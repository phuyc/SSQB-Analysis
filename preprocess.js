/**
 * @fileoverview Processes reading question data by removing HTML tags from text content
 * @module ssqb-analysis/test
 * 
 * @typedef {Object} AnswerOption
 * @property {string} id - The unique identifier for the answer option
 * @property {string} content - The text content of the answer option
 * 
 * @typedef {Object} ReadingQuestion
 * @property {string} id - Unique identifier for the question
 * @property {string} template - Template identifier
 * @property {string} question - Question text
 * @property {string} answer - Correct answer text
 * @property {string[]} distractors - Array of incorrect answer options
 * @property {string} vaultid - Vault identifier
 * @property {string[]} keys - Array of associated keys/tags
 * @property {string} rationale - Explanation of the correct answer
 * @property {string} origin - Source of the question
 * @property {string} stem - Question stem
 * @property {string} externalid - External identifier
 * @property {string} stimulus - Reading passage or context
 * @property {string} templateclusterid - Template cluster identifier
 * @property {string} parenttemplatename - Name of parent template
 * @property {string} parenttemplateid - ID of parent template
 * @property {string} type - Question type
 * @property {string} position - Position in sequence
 * @property {string} templateclustername - Name of template cluster
 * @property {AnswerOption[]} answerOptions - Array of possible answers
 * @property {string[]} correct_answer - Array of correct answer identifiers
 * 
 * @function getTextContent
 * @param {string} string - Input string containing HTML tags
 * @returns {string} Text content with HTML tags removed
 * 
 * The script reads a JSON file containing reading questions, removes HTML tags 
 * from text fields (stimulus, question, rationale, and answer options), adds metadata,
 * and writes the processed data to a new JSON file.
 */
import fs from 'fs';
import _ from 'lodash';

/**
 * 
 * @param {string} string - Input string containing HTML tags 
 * @returns {string} String with table tags removed
 */
function removeTable(string) {
    return string?.replace(/<table[^>]*>.*?<\/table>/gs, '');
}

/**
 * Extracts text content from a string by removing HTML tags
 * @param {string} string - Input string containing HTML tags
 * @returns {string} Text content with HTML tags removed
*/
function getTextContent(string) {
    return string?.replace(/<[^>]*>/g, '');
}

/**
 * Replaces HTML entities with their corresponding characters
 * @param {string} str - Input string with HTML entities
 * @returns {string} String with HTML entities replaced by characters
 */
function replaceEntities(str) {
    const entities = {
      '&rsquo;': '’',
      '&lsquo;': '‘',
      '&rdquo;': '”',
      '&ldquo;': '“',
      '&ndash;': '–',
      '&mdash;': '—',
      '&nbsp;': ' ',
      '&amp;': '&',
      '&lt;': '<',
      '&gt;': '>',
      '&deg;': '°',
    };
    return str?.replace(/&[a-z]+;/g, match => entities[match]);
}

try {
    const data = fs.readFileSync('all-detailed-reading-questions.json', 'utf8');
    const metadata = fs.readFileSync('all-reading-questions.json', 'utf8');
    
    /**
     * @type {ReadingQuestion[]}
     * */
    let jsonArray = JSON.parse(data);
    let metadataArray = JSON.parse(metadata);

    jsonArray = jsonArray.map((item) => {
        // Process the JSON data by removing HTML tags from text fields
        item.stimulus = replaceEntities(getTextContent(item.stimulus));
        item.question = replaceEntities(getTextContent(item.question));
        item.rationale = replaceEntities(getTextContent(item.rationale));
        item.stem = replaceEntities(getTextContent(item.stem));
        item.answerOptions = item.answerOptions.map((answerOption) => {
            answerOption.content = replaceEntities(getTextContent(answerOption.content));
            return answerOption;
        });

        // Add metadata to each question
        const metadataItem = metadataArray.find((metadataItem) => metadataItem.external_id === item.externalid);
        if (metadataItem) {
            return { ...item, ...metadataItem };
        }

        return item;
    });

    fs.writeFileSync('test.json', JSON.stringify(jsonArray, null, 2));
} catch (error) {
    console.error('Error reading file:', error);
}