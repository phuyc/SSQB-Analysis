let allXHRResponses;

// Save the original XMLHttpRequest open method
const originalXhrOpen = XMLHttpRequest.prototype.open;

XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    // Add an event listener to capture the response when the request completes
    this.addEventListener('load', function () {
        // Store the response data
        allXHRResponses = this.responseText;
        console.log("Captured response from:", url);
        saveAllXHRResponses();
    });

    // Call the original open method to ensure the request proceeds
    originalXhrOpen.call(this, method, url, ...rest);
};

// Save all collected responses to a file
function saveAllXHRResponses() {
    const blob = new Blob([allXHRResponses], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'all-questions.json';
    link.click();
    console.log("Saved all questions to file");
}
