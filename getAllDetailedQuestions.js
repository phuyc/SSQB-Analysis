let allXHRResponses = [];

// Save the original XMLHttpRequest open method
const originalXhrOpen = XMLHttpRequest.prototype.open;

XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    // Add an event listener to capture the response when the request completes
    this.addEventListener('load', function () {
        // Store the response data
        allXHRResponses.push(this.responseText);
        console.log("Captured response from:", url);
    });
    // Call the original open method to ensure the request proceeds
    originalXhrOpen.call(this, method, url, ...rest);
};

// The number of questions to collect
const numQuestions = document.getElementById("results-table").getElementsByTagName("span")[0].textContent;
const pages = Math.floor(numQuestions / 10) + 1;
const next16 = document.getElementById("undefined_next");

TickAndClick(next16, 0);

function dsat16autotick() {
    const questionButtons = document.querySelectorAll('.view-question-button');
    
    for (const button of questionButtons) {
        button.click();
    }
}

// Save all collected responses to a file
function saveAllXHRResponses() {
    const blob = new Blob(JSON.parse(JSON.stringify(allXHRResponses)), { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'xhr_responses.json';
    link.click();
    console.log("Saved all XHR responses to file!");
}

async function TickAndClick(node, i) {
    setTimeout(async () => {
        if (i < pages) {
            dsat16autotick();
            node.click();
            await TickAndClick(node, i + 1);
        } else {     
            saveAllXHRResponses();     
        }
    }, 300)
}