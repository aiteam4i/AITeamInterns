document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const pdfFilesInput = document.getElementById('pdfFiles');
    const uploadStatus = document.getElementById('uploadStatus');
    const uploadedFilesList = document.getElementById('uploadedFilesList');
    const uploadedCount = document.getElementById('uploadedCount');
    const generateGraphBtn = document.getElementById('generateGraphBtn');
    const graphStatus = document.getElementById('graphStatus');
    const chatForm = document.getElementById('chatForm');
    const userQuestionInput = document.getElementById('userQuestion');
    const chatContainer = document.getElementById('chatContainer');
    const sendBtn = document.getElementById('sendBtn');
    const sendText = document.getElementById('sendText');
    const spinner = document.getElementById('spinner');

    // Chat history stored in JavaScript for this no-session setup
    let chatHistory = []; 

    // Function to render chat messages
    function renderChatHistory() {
        chatContainer.innerHTML = ''; // Clear existing messages
        chatHistory.forEach(message => {
            appendMessage(message.content, message.role, false); // Don't scroll immediately
        });
        chatContainer.scrollTop = chatContainer.scrollHeight; // Scroll to bottom once
    }

    // Append a new message and optionally scroll
    function appendMessage(text, role, shouldScroll = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        bubbleDiv.textContent = text;
        messageDiv.appendChild(bubbleDiv);
        chatContainer.appendChild(messageDiv);
        if (shouldScroll) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    // Initial render of chat history (empty in this no-session setup, unless you manually add some defaults)
    renderChatHistory();

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        uploadStatus.textContent = 'Uploading and processing...';
        uploadStatus.className = 'status-message'; // Reset class

        const formData = new FormData();
        for (const file of pdfFilesInput.files) {
            formData.append('pdf_files', file);
        }

        try {
            const response = await fetch('/upload_pdf', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.status === 'success') {
                uploadStatus.textContent = data.message;
                uploadStatus.classList.add('success');
                uploadedFilesList.innerHTML = ''; // Clear existing list
                data.uploaded_files.forEach(file => {
                    const li = document.createElement('li');
                    li.textContent = file.name;
                    uploadedFilesList.appendChild(li);
                });
                uploadedCount.textContent = data.uploaded_files.length;
            } else {
                uploadStatus.textContent = data.message;
                uploadStatus.classList.add('error');
            }
        } catch (error) {
            uploadStatus.textContent = `Error: ${error.message}`;
            uploadStatus.classList.add('error');
            console.error('Upload error:', error);
        }
    });

    generateGraphBtn.addEventListener('click', async function() {
        graphStatus.textContent = 'Generating graph...';
        graphStatus.className = 'status-message'; // Reset class
        generateGraphBtn.disabled = true; // Disable button during processing

        try {
            const response = await fetch('/generate_graph', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            const data = await response.json();

            if (data.status === 'success') {
                graphStatus.textContent = data.message;
                graphStatus.classList.add('success');
            } else if (data.status === 'warning') {
                graphStatus.textContent = data.message;
                graphStatus.classList.add('warning');
            } else {
                graphStatus.textContent = data.message;
                graphStatus.classList.add('error');
            }
        } catch (error) {
            graphStatus.textContent = `Error: ${error.message}`;
            graphStatus.classList.add('error');
            console.error('Generate graph error:', error);
        } finally {
            generateGraphBtn.disabled = false; // Re-enable button
        }
    });

    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const question = userQuestionInput.value.trim();
        if (!question) return;

        // Add user message to history and display
        chatHistory.push({ role: 'user', content: question });
        appendMessage(question, 'user');
        userQuestionInput.value = '';

        sendBtn.disabled = true;
        sendText.style.display = 'none';
        spinner.style.display = 'inline-block';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: question })
            });
            const data = await response.json();

            // Add assistant message to history and display
            chatHistory.push({ role: 'assistant', content: data.response });
            appendMessage(data.response, 'assistant');
            
        } catch (error) {
            // Handle network errors or server issues
            const errorMessage = `An error occurred: ${error.message}. Please check server logs.`;
            chatHistory.push({ role: 'assistant', content: errorMessage });
            appendMessage(errorMessage, 'assistant');
            console.error('Chat error:', error);
        } finally {
            sendBtn.disabled = false;
            sendText.style.display = 'inline';
            spinner.style.display = 'none';
            chatContainer.scrollTop = chatContainer.scrollHeight; // Ensure scroll to bottom
        }
    });
});