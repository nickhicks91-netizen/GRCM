// Children's Book PDF Generator
// All processing happens client-side in the browser

document.addEventListener('DOMContentLoaded', init);

// Initialize jsPDF
const { jsPDF } = window.jspdf;

// Element selectors
let trimSizeSelect;
let marginInput;
let bleedCheckbox;
let manuscriptUpload;
let imageUpload;
let generateBtn;
let statusMessage;

/**
 * Initialize the application
 */
function init() {
    // Get references to all interactive elements
    trimSizeSelect = document.getElementById('trim-size');
    marginInput = document.getElementById('margin-input');
    bleedCheckbox = document.getElementById('bleed-checkbox');
    manuscriptUpload = document.getElementById('manuscript-upload');
    imageUpload = document.getElementById('image-upload');
    generateBtn = document.getElementById('generate-btn');
    statusMessage = document.getElementById('status-message');

    // Add event listener to the generate button
    generateBtn.addEventListener('click', buildBook);
}

/**
 * Main function to build the PDF book
 */
async function buildBook() {
    try {
        // Clear previous status
        setStatus('Gathering settings...', 'info');

        // Get settings
        const trimSizeValue = trimSizeSelect.value;
        const [widthStr, heightStr] = trimSizeValue.split('x');
        const width = parseFloat(widthStr);
        const height = parseFloat(heightStr);
        const margin = parseFloat(marginInput.value);
        const bleed = bleedCheckbox.checked;

        // Get files
        const manuscriptFile = manuscriptUpload.files[0];
        const imageFiles = Array.from(imageUpload.files);

        // Validation
        if (!manuscriptFile) {
            setStatus('Error: Please upload a manuscript file.', 'error');
            return;
        }

        if (imageFiles.length === 0) {
            setStatus('Error: Please upload at least one illustration.', 'error');
            return;
        }

        // Disable button during processing
        generateBtn.disabled = true;

        // Read files
        setStatus('Loading files...', 'info');

        const manuscriptText = await manuscriptFile.text();
        const imageDataPromises = imageFiles.map(file => readFileAsDataURL(file));
        const imageDataURLs = await Promise.all(imageDataPromises);

        // Parse manuscript
        setStatus('Parsing manuscript...', 'info');

        // Split manuscript by double line breaks
        const textPages = manuscriptText.split('\n\n').filter(page => page.trim().length > 0);

        // Validate page count
        if (textPages.length !== imageFiles.length) {
            setStatus(
                `Error: Manuscript has ${textPages.length} pages but ${imageFiles.length} images were uploaded. They must match.`,
                'error'
            );
            generateBtn.disabled = false;
            return;
        }

        // Initialize PDF
        setStatus('Initializing PDF...', 'info');
        const doc = new jsPDF({
            unit: 'in',
            format: [width, height]
        });

        // Process each page
        for (let i = 0; i < textPages.length; i++) {
            setStatus(`Processing page ${i + 1}/${textPages.length}...`, 'info');

            // Add new page (except for the first one)
            if (i > 0) {
                doc.addPage();
            }

            // Load image and get dimensions
            const imageData = await loadImage(imageDataURLs[i]);

            // Add image to page
            await addImageToPage(doc, imageData, width, height, margin, bleed);

            // Add text to page
            addTextToPage(doc, textPages[i], width, height, margin);
        }

        // Save PDF
        setStatus('Finalizing PDF...', 'info');
        doc.save('my-childrens-book.pdf');

        setStatus('Done! Your book has been downloaded.', 'success');
        generateBtn.disabled = false;

    } catch (error) {
        console.error('Error generating PDF:', error);
        setStatus(`Error: ${error.message}`, 'error');
        generateBtn.disabled = false;
    }
}

/**
 * Read a file as a Data URL
 * @param {File} file - The file to read
 * @returns {Promise<string>} - Promise that resolves with the data URL
 */
function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (e) => {
            resolve(e.target.result);
        };

        reader.onerror = (e) => {
            reject(new Error(`Failed to read file: ${file.name}`));
        };

        reader.readAsDataURL(file);
    });
}

/**
 * Load an image and get its dimensions
 * @param {string} dataURL - The image data URL
 * @returns {Promise<Object>} - Promise that resolves with image data and dimensions
 */
function loadImage(dataURL) {
    return new Promise((resolve, reject) => {
        const img = new Image();

        img.onload = () => {
            resolve({
                dataURL: dataURL,
                width: img.naturalWidth,
                height: img.naturalHeight
            });
        };

        img.onerror = () => {
            reject(new Error('Failed to load image'));
        };

        img.src = dataURL;
    });
}

/**
 * Add image to PDF page with proper sizing
 * @param {jsPDF} doc - The PDF document
 * @param {Object} imageData - Image data with dataURL, width, height
 * @param {number} pageWidth - Page width in inches
 * @param {number} pageHeight - Page height in inches
 * @param {number} margin - Margin in inches
 * @param {boolean} bleed - Whether to use full bleed
 */
function addImageToPage(doc, imageData, pageWidth, pageHeight, margin, bleed) {
    const imgAspectRatio = imageData.width / imageData.height;

    let x, y, imgWidth, imgHeight;

    if (bleed) {
        // Full bleed: fill entire page, potentially cropping
        const pageAspectRatio = pageWidth / pageHeight;

        if (imgAspectRatio > pageAspectRatio) {
            // Image is wider than page - fit to height
            imgHeight = pageHeight;
            imgWidth = imgHeight * imgAspectRatio;
            x = (pageWidth - imgWidth) / 2;
            y = 0;
        } else {
            // Image is taller than page - fit to width
            imgWidth = pageWidth;
            imgHeight = imgWidth / imgAspectRatio;
            x = 0;
            y = (pageHeight - imgHeight) / 2;
        }
    } else {
        // No bleed: fit image within margins
        const availableWidth = pageWidth - (margin * 2);
        const availableHeight = pageHeight - (margin * 2);
        const availableAspectRatio = availableWidth / availableHeight;

        if (imgAspectRatio > availableAspectRatio) {
            // Image is wider - fit to available width
            imgWidth = availableWidth;
            imgHeight = imgWidth / imgAspectRatio;
            x = margin;
            y = margin + (availableHeight - imgHeight) / 2;
        } else {
            // Image is taller - fit to available height
            imgHeight = availableHeight;
            imgWidth = imgHeight * imgAspectRatio;
            x = margin + (availableWidth - imgWidth) / 2;
            y = margin;
        }
    }

    // Add image to PDF
    doc.addImage(imageData.dataURL, 'JPEG', x, y, imgWidth, imgHeight);
}

/**
 * Add text to PDF page
 * @param {jsPDF} doc - The PDF document
 * @param {string} text - The text to add
 * @param {number} pageWidth - Page width in inches
 * @param {number} pageHeight - Page height in inches
 * @param {number} margin - Margin in inches
 */
function addTextToPage(doc, text, pageWidth, pageHeight, margin) {
    // Set font properties
    doc.setFontSize(14);
    doc.setTextColor(0, 0, 0);

    // Calculate text area dimensions
    const textWidth = pageWidth - (margin * 2);
    const textX = margin;

    // Position text at the bottom of the page
    // We'll place it near the bottom, leaving some space
    const textY = pageHeight - margin - 0.5;

    // Split text into lines that fit within the width
    const lines = doc.splitTextToSize(text.trim(), textWidth);

    // Calculate total text height
    const lineHeight = 0.2; // inches
    const totalTextHeight = lines.length * lineHeight;

    // Adjust Y position to account for multiple lines (move up from bottom)
    const adjustedY = textY - totalTextHeight + lineHeight;

    // Add text with wrapping
    doc.text(lines, textX, adjustedY, {
        maxWidth: textWidth,
        align: 'left'
    });
}

/**
 * Set status message
 * @param {string} message - The message to display
 * @param {string} type - The type of message ('info', 'success', 'error')
 */
function setStatus(message, type = 'info') {
    statusMessage.textContent = message;
    statusMessage.className = '';

    if (type === 'success') {
        statusMessage.classList.add('success');
    } else if (type === 'error') {
        statusMessage.classList.add('error');
    }
}
