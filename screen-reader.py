import time
import pyautogui
import keyboard
import pytesseract
from PIL import ImageGrab, ImageDraw
import re
import json
import os
import pandas as pd
import winsound
import cv2
import numpy as np

VERBOSE = True

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class RegionCapture:
    def __init__(self, config_file='region_config.json'):
        self.config_file = config_file
        self.region = self.load_region()

    def load_region(self):
        """Load saved region coordinates from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None

    def save_region(self, region):
        """Save region coordinates to file"""
        with open(self.config_file, 'w') as f:
            json.dump(region, f)
        self.region = region

    def capture_new_region(self):
        """Capture new region coordinates"""
        print("Position your mouse at the TOP-LEFT corner and press 'Enter'.")
        while not keyboard.is_pressed("enter"):
            pass
        start_x, start_y = pyautogui.position()
        print(f"Start position: ({start_x}, {start_y})")
        time.sleep(0.5)

        print("Now position your mouse at the BOTTOM-RIGHT corner and press 'Enter'.")
        while not keyboard.is_pressed("enter"):
            pass
        end_x, end_y = pyautogui.position()
        print(f"End position: ({end_x}, {end_y})")
        time.sleep(0.5)

        region = {
            'left': min(start_x, end_x),
            'top': min(start_y, end_y),
            'right': max(start_x, end_x),
            'bottom': max(start_y, end_y)
        }
        self.save_region(region)
        return region

    def capture_screen(self):
        """Capture the screen using saved or new region"""
        if not self.region:
            print("No saved region found. Please select a new region.")
            self.region = self.capture_new_region()
        
        screenshot = ImageGrab.grab(bbox=(
            self.region['left'],
            self.region['top'],
            self.region['right'],
            self.region['bottom']
        ))
        return screenshot

def preprocess_image(img):
    """
    Preprocess the image to improve OCR accuracy.
    """
    # Convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')
    
    # Increase contrast
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    return img

def is_header_text(text):
    """
    Check if the text is part of the header (question number or "Mark for Review")
    """
    # Check for question number pattern (1-9999)
    if re.match(r'^\d{1,4}$', text.strip()):
        return True
    
    # Check for "Mark for Review" text
    if 'mark' in text.lower() and 'review' in text.lower():
        return True
        
    return False

def detect_question_header(img):
    """
    Detect the question number and 'Mark for Review' header more accurately.
    Returns y-coordinate of the bottom of the header.
    """
    # Only look at top 20% of the image for header
    header_region = img.crop((0, 0, img.width, int(img.height * 0.2)))
    data = pytesseract.image_to_data(header_region, output_type=pytesseract.Output.DICT)
    
    header_elements = []
    
    # Find all header elements
    for i, text in enumerate(data['text']):
        if text.strip() and int(data['conf'][i]) > 0:
            if is_header_text(text):
                y_bottom = int(data['top'][i]) + int(data['height'][i])
                header_elements.append(y_bottom)
    
    if not header_elements:
        # If no header elements found, use a minimal header height
        return 30  # Default minimum header height
    
    # Use the bottom of the lowest header element
    return max(header_elements) + 5  # Add small padding

def detect_question_text(img, header_bottom):
    """
    More accurate question text detection
    """
    answers_start = find_first_answer_y(img, header_bottom)
    
    # Use a slightly smaller region to avoid catching answer text
    question_region = img.crop((0, header_bottom, img.width, answers_start - 20))
    question_text = pytesseract.image_to_string(preprocess_image(question_region))
    
    # Clean the question text
    question_text = re.sub(r'[®©°™\(\)@\[\]\{\}<>]', '', question_text)
    question_text = ' '.join(question_text.split())
    
    # Verify it doesn't look like an answer choice
    if re.match(r'^[A-D][\.:\)\}]', question_text):
        return ""
        
    return question_text.strip()

def clean_answer_text(text):
    """
    More thorough cleaning of answer text
    """
    # Remove all special characters and symbols
    text = re.sub(r'[®©°™\(\)@\[\]\{\}<>]', '', text)
    # Remove potential answer labels at start
    text = re.sub(r'^[A-Da-d][\.:\)\}]\s*', '', text)
    text = text.replace("|", "")
    # Remove extra spaces and normalize whitespace
    text = ' '.join(text.split())
    return text.strip()

def detect_answer_boxes(img):
    """
    Enhanced box detection with better thresholding and edge detection
    """
    # Convert PIL image to cv2 format
    cv_img = np.array(img)
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv_img

    # Apply multiple processing steps to better detect boxes
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find rectangles that are likely answer boxes
    answer_boxes = []
    min_width = img.width * 0.10  # Boxes should be at least 10% of image width
    
    for contour in contours:
        # Approximate the contour to remove noise
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # More precise filtering criteria
        if (2.5 < aspect_ratio and    # Answer boxes are typically wide
            w > min_width and              # Must be sufficiently wide
            45 < h < 150 and               # Typical height range for answer text
            len(approx) >= 4):            # Should be roughly rectangular
            answer_boxes.append((y, y+h, x + 60, x+w))
    
    # Sort boxes by vertical position
    answer_boxes.sort()
    return answer_boxes

def find_first_answer_y(img, header_bottom):
    """
    Find answer section using box detection
    """
    # Look for answer boxes in the region below header
    answer_region = img.crop((0, header_bottom, img.width, img.height))
    boxes = detect_answer_boxes(answer_region)
    
    if boxes:
        # Return y-coordinate of first box, adjusted for header offset
        return header_bottom + max(0, boxes[0][0] - 5)  # Small padding above first box
    
    # Fallback to previous method if no boxes found
    return int(img.height * 0.4)

def detect_answer_choices(img, header_bottom):
    """
    Enhanced answer choice detection with better error handling
    """
    answer_region = img.crop((0, header_bottom, img.width, img.height))
    boxes = detect_answer_boxes(answer_region)
    
    if VERBOSE:
        print(f"Found {len(boxes)} potential answer boxes")
    
    answers = []
    for box_top, box_bottom, box_left, box_right in boxes[:4]:
        # Add padding to ensure we capture all text
        padding = 2
        box_region = answer_region.crop((
            max(0, box_left - padding),
            max(0, box_top - padding),
            min(answer_region.width, box_right + padding),
            min(answer_region.height, box_bottom + padding)
        ))
        
        # Enhance the box region for better OCR
        box_region = preprocess_image(box_region)
        
        text = pytesseract.image_to_string(box_region)
        cleaned_text = clean_answer_text(text)
        
        if cleaned_text and len(cleaned_text) > 1:
            answers.append(cleaned_text)
    
    return answers

def find_and_add_blanks(text):
    """
    Find potential blank spaces in the text that need to be filled in.
    Then, add a placeholder (e.g., '_____') to indicate the blank.
    """
    # Pattern matches:
    # - One or more underscores
    # - Two or more dashes
    # - Two or more spaces between words
    # - Special blank line characters OCR might detect
    blank_patterns = [
        r'[_]{1,}',         # One or more underscores
        r'[-]{2,}',         # Two or more hyphens
        r'[—]{2,}',         # Two or more em dashes
        r'\s{2,}',          # Two or more spaces
        r'[\u2000-\u200F]', # Various Unicode whitespace characters
        r'[\u2028-\u202F]'  # More Unicode whitespace/special characters
    ]
    
    combined_pattern = '|'.join(blank_patterns)
    
    # Find all matches in the textz
    matches = re.finditer(combined_pattern, text)
    for match in matches:
        # Replace the match with underscores
        text.replace(match.group(), '_____')
        
    return text

def play_ready_sound():
    """Play a brief beep to indicate ready state"""
    frequency = 1000  # Set frequency to 1000 Hz
    duration = 200    # Set duration to 200 ms
    winsound.Beep(frequency, duration)

def main():
    region_capture = RegionCapture()
    print("=== Enhanced Test Question Reader ===")
    print("Press 'n' to set a new region")
    print("Press 'Enter' to capture the current region")
    print("Press 'q' to quit")

    list = []

    while True:
        if keyboard.is_pressed('n'):
            region_capture.capture_new_region()
            time.sleep(0.5)
            print("\nNew region saved. Press 'Enter' to capture or 'q' to quit.")
        
        elif keyboard.is_pressed('enter'):
            try:
                full_img = region_capture.capture_screen()
                # Process the image and display results
                w, h = full_img.size
    
                # Split into passage (left) and question (right) sections
                passage_img = full_img.crop((0, 0, w//2, h))
                question_img = full_img.crop((w//2, 0, w, h))
                
                # Preprocess images
                passage_img = preprocess_image(passage_img)
                question_img = preprocess_image(question_img)
                
                # Process question section
                header_bottom = detect_question_header(question_img)
                question_text = detect_question_text(question_img, header_bottom)
                answer_choices = detect_answer_choices(question_img, header_bottom)

                # Extract passage text
                passage_text = pytesseract.image_to_string(passage_img)

                if VERBOSE:
                    print("\nPassage:")
                    print(find_and_add_blanks(passage_text))
                    
                    print("\nQuestion:")
                    print(question_text)
                    
                    print("\nAnswer Choices:")
                    for i, text in enumerate(['A', 'B', 'C', 'D']):
                        if i < len(answer_choices):
                            print(f"{text}. {answer_choices[i]}")

                print("Captured question #", len(list) + 1)

                # Save data to DataFrame
                list.append({
                    'Stimulus': passage_text,
                    'Stem': question_text,
                    'answerOptions': answer_choices
                })

                print("Saved to DataFrame")

                # Save processed regions for debugging
                debug_img = full_img.copy()
                draw = ImageDraw.Draw(debug_img)
                
                # Draw split lines
                draw.line([(w//2, 0), (w//2, h)], fill="red", width=2)
                draw.line([(w//2, header_bottom), (w, header_bottom)], fill="blue", width=2)
                
                # Draw answer choice boxes - fixed coordinate mapping
                answer_boxes = detect_answer_boxes(question_img)
                
                for box_top, box_bottom, box_left, box_right in answer_boxes:
                    # Adjust coordinates relative to the full image:
                    # 1. Add w//2 to x-coordinates (right half offset)
                    # 2. Add header_bottom to y-coordinates (question section offset)
                    adjusted_coords = [
                        w//2 + box_left,           # left
                        box_top,   # top
                        w//2 + box_right,          # right
                        box_bottom  # bottom
                    ]
                    
                    # Draw rectangle with corrected coordinates
                    draw.rectangle(
                        [(adjusted_coords[0], adjusted_coords[1]), 
                         (adjusted_coords[2], adjusted_coords[3])],
                        outline="green",
                        width=2
                    )
                                    
                debug_img.save("processed_regions.png")
                print("\nDebug image saved as 'processed_regions.png'")

                print("\nPress 'Enter' for next capture, 'n' for new region, 's' to save to csv, or 'q' to quit.")
                time.sleep(0.5)
                play_ready_sound()
            except Exception as e:
                print(f"Error during capture: {e}")
        elif keyboard.is_pressed('s'):
            # Save DataFrame to CSV
            df = pd.DataFrame(list, columns=['Stimulus', 'Stem', 'answerOptions'])
            df.to_json('new_practice_tests.json', index=False)
            print("Saved DataFrame to 'questions.csv'")
        elif keyboard.is_pressed('q'):
            print("Exiting...")
            break
        
        time.sleep(0.1)  # Prevent high CPU usage

if __name__ == "__main__":
    main()
