// =========================
// MARKED CONFIG
// =========================

marked.setOptions({
    breaks: true,
    gfm: true
});

let sessionId = localStorage.getItem("session_id");
let sessions = JSON.parse(localStorage.getItem("sessions") || "{}");

// =========================
// DOM READY
// =========================

document.addEventListener("DOMContentLoaded", function () {

    const sendBtn = document.getElementById("sendBtn");
    const textarea = document.getElementById("message");

    if (sendBtn) sendBtn.addEventListener("click", sendMessage);

    const newChatBtn = document.querySelector(".new-chat");
    if (newChatBtn) newChatBtn.addEventListener("click", newChat);

    if (textarea) {
        textarea.addEventListener("keydown", function (e) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        textarea.addEventListener("input", () => {
            textarea.style.height = "auto";
            textarea.style.height = textarea.scrollHeight + "px";
        });
    }

    renderHistory();

    if (sessionId && sessions[sessionId]) {
        loadSession(sessionId);
    } else {
        showNewChatWelcome();
    }

    focusInput();
});

// =========================
// FOCUS
// =========================

function focusInput() {
    const input = document.getElementById("message");
    if (!input) return;
    input.focus();
    const len = input.value.length;
    input.setSelectionRange(len, len);
}

// =========================
// NEW CHAT
// =========================

function newChat() {
    sessionId = null;
    localStorage.removeItem("session_id");
    renderHistory();
    showNewChatWelcome();
    focusInput();
}

// =========================
// WELCOME
// =========================

function showNewChatWelcome() {
    const chat = document.getElementById("chat-box");
    if (!chat) return;

    chat.innerHTML = "";

    const welcome = document.createElement("div");
    welcome.className = "chat-welcome";
    welcome.innerHTML = `
        <h2>👋 Новый диалог</h2>
        <p>Задайте вопрос по документации</p>
        <p>Я отвечу строго на основе найденных данных</p>
    `;
    chat.appendChild(welcome);
}

// =========================
// SEND MESSAGE
// =========================

async function sendMessage() {

    const input = document.getElementById("message");
    if (!input) return;

    const message = input.value.trim();
    if (!message) return;

    const welcome = document.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    if (!sessionId || !sessions[sessionId]) {
        sessionId = "session_" + Date.now();
        sessions[sessionId] = {
            title: message.substring(0, 40),
            messages: [],
            created: Date.now()
        };
        localStorage.setItem("session_id", sessionId);
    }

    // USER MESSAGE
    sessions[sessionId].messages.push({
        role: "user",
        content: message
    });

    saveSessions();
    renderHistory();

    appendMessage("user", message);

    input.value = "";
    input.style.height = "auto";

    // ASSISTANT PLACEHOLDER
    sessions[sessionId].messages.push({
        role: "assistant",
        content: "",
        html: "",
        sources: [],
        confidence: null
    });

    saveSessions();

    const assistantMessage = appendMessage("assistant", "");
    const contentDiv =
        assistantMessage.querySelector(".assistant-content");

    // =========================
    // TYPING INDICATOR
    // =========================
    const typingIndicator = document.createElement("div");
    typingIndicator.className = "typing";
    typingIndicator.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    contentDiv.appendChild(typingIndicator);

    try {

        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-Session-Id": sessionId
            },
            body: JSON.stringify({ question: message })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let fullText = "";
        let contentText = "";
        let sources = [];
        let confidence = null;

        while (true) {

            const { done, value } = await reader.read();
            if (done) break;

            fullText += decoder.decode(value);

            const confSplit = fullText.split("###CONFIDENCE###");
            const mainPart = confSplit[0];

            if (confSplit.length > 1) {
                confidence = parseFloat(confSplit[1].trim());
            }

            const sourceSplit = mainPart.split("###SOURCES###");
            contentText = sourceSplit[0];

            if (sourceSplit.length > 1) {
                try {
                    sources = JSON.parse(sourceSplit[1].trim());
                } catch (e) {}
            }

            // Убираем typing indicator при первом токене
            const typing = contentDiv.querySelector(".typing");
            if (typing) typing.remove();

            // =========================
            // ОБРАБОТКА "НЕ НАЙДЕНО"
            // =========================
            if (contentText.trim() ===
                "В документации информация не найдена.") {

                contentDiv.innerHTML = `
                    <div class="no-results">
                        <div class="no-results-icon">🔍</div>
                        <div class="no-results-title">
                            Информация не найдена
                        </div>
                        <div class="no-results-text">
                            Попробуйте:
                            <ul>
                                <li>Уточнить формулировку</li>
                                <li>Добавить больше контекста</li>
                                <li>Разбить вопрос на части</li>
                            </ul>
                        </div>
                    </div>
                `;
            } else {

                contentDiv.innerHTML =
                    marked.parse(contentText);

                // оптимизированная подсветка
                contentDiv
                    .querySelectorAll("pre code")
                    .forEach(block => {
                        hljs.highlightElement(block);
                    });
            }

            let lastMsg =
                sessions[sessionId].messages[
                    sessions[sessionId].messages.length - 1
                ];

            lastMsg.content = contentText;
            lastMsg.html = contentDiv.innerHTML;

            saveSessions();
        }

        // =========================
        // ДОБАВЛЯЕМ WARNING + META
        // =========================

        let lastMsg =
            sessions[sessionId].messages[
                sessions[sessionId].messages.length - 1
            ];

        lastMsg.sources = sources;
        lastMsg.confidence = confidence;

        // ⚠ Предупреждение при низкой уверенности
        if (confidence !== null &&
            confidence < 0.5 &&
            contentDiv) {

            const warning =
                document.createElement("div");
            warning.className = "low-confidence-warning";
            warning.innerHTML = `
                ⚠ Ответ может быть неточным.
                Попробуйте уточнить формулировку запроса.
            `;
            contentDiv.appendChild(warning);
        }

        // Блок источников и confidence
        if ((sources.length > 0 ||
             confidence !== null) &&
             contentDiv) {

            const metaBlock =
                document.createElement("div");
            metaBlock.className = "sources";

            let html = "";

            if (sources.length > 0) {
                html += "<strong>Источники:</strong><br>";
                html += sources.map(s =>
                    `• ${s}`).join("<br>");
            }

            if (confidence !== null &&
                confidence !== undefined) {

                const percent =
                    Math.round(confidence * 100);

                let levelClass = "conf-low";
                if (percent >= 80)
                    levelClass = "conf-high";
                else if (percent >= 60)
                    levelClass = "conf-medium";

                html += `
                    <div class="confidence ${levelClass}">
                        Уверенность: ${percent}%
                    </div>
                `;
            }

            metaBlock.innerHTML = html;
            contentDiv.appendChild(metaBlock);
        }

        lastMsg.html = contentDiv.innerHTML;
        saveSessions();

        addCopyButton(assistantMessage);

        // =========================
        // АВТОГЕНЕРАЦИЯ ЗАГОЛОВКА
        // =========================
        if (sessions[sessionId] &&
            sessions[sessionId].messages &&
            sessions[sessionId].messages.length === 2) {

            const question =
                sessions[sessionId].messages[0].content;

            const answer =
                sessions[sessionId].messages[1].content;

            fetch("/generate-title", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question: question,
                    answer: answer
                })
            })
            .then(res => {
                if (!res.ok) return null;
                return res.json();
            })
            .then(data => {
                if (data && data.title) {
                    sessions[sessionId].title =
                        data.title;
                    saveSessions();
                    renderHistory();
                }
            })
            .catch(() => {});
        }

    } catch (err) {
        console.error("Chat error:", err);
    }

    focusInput();
}

// =========================
// APPEND MESSAGE
// =========================

function appendMessage(role, text) {

    const chat = document.getElementById("chat-box");
    if (!chat) return;

    const div = document.createElement("div");
    div.className = `message ${role}`;

    if (role === "assistant") {
        const contentDiv = document.createElement("div");
        contentDiv.className = "assistant-content";
        contentDiv.innerHTML = text;
        div.appendChild(contentDiv);
    } else {
        div.innerText = text;
    }

    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;

    return div;
}

// =========================
// LOAD SESSION
// =========================

function loadSession(id) {

    sessionId = id;
    localStorage.setItem("session_id", id);

    const chat = document.getElementById("chat-box");
    if (!chat) return;

    chat.innerHTML = "";

    if (!sessions[id] || !sessions[id].messages) return;

    sessions[id].messages.forEach(msg => {

        let messageElement;

        if (msg.role === "assistant") {

            messageElement = appendMessage("assistant", "");
            const contentDiv =
                messageElement.querySelector(".assistant-content");

            if (contentDiv) {

                if (msg.html && msg.html.trim() !== "") {
                    contentDiv.innerHTML = msg.html;
                } else {
                    contentDiv.innerHTML =
                        marked.parse(msg.content || "");
                }

                contentDiv
                    .querySelectorAll("pre code")
                    .forEach(block => {
                        hljs.highlightElement(block);
                    });
            }

            addCopyButton(messageElement);

        } else {

            messageElement =
                appendMessage("user", msg.content);
        }
    });

    chat.scrollTop = chat.scrollHeight;
    renderHistory();
    focusInput();
}

// =========================
// COPY BUTTON
// =========================

function addCopyButton(assistantMessage) {

    const contentDiv =
        assistantMessage.querySelector(".assistant-content");
    if (!contentDiv) return;

    if (assistantMessage.querySelector(".copy-icon")) return;

    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-icon";
    copyBtn.innerHTML = "📋";

    copyBtn.onclick = () => {

        const clone = contentDiv.cloneNode(true);
        const sourcesBlock = clone.querySelector(".sources");
        if (sourcesBlock) sourcesBlock.remove();

        navigator.clipboard.writeText(clone.innerText);

        copyBtn.innerHTML = "✓";
        setTimeout(() => {
            copyBtn.innerHTML = "📋";
        }, 1500);
    };

    assistantMessage.appendChild(copyBtn);
}

// =========================
// HISTORY
// =========================

function renderHistory() {

    const history = document.getElementById("history");
    if (!history) return;

    history.innerHTML = "";

    Object.entries(sessions)
        .sort((a, b) => b[1].created - a[1].created)
        .forEach(([id, session]) => {

            const item = document.createElement("div");
            item.className = "history-item";
            if (id === sessionId) item.classList.add("active");

            const title = document.createElement("span");
            title.className = "history-title";
            title.innerText = session.title;

            // DOUBLE CLICK RENAME
            title.ondblclick = (e) => {
                e.stopPropagation();

                const input = document.createElement("input");
                input.type = "text";
                input.value = session.title;
                input.className = "rename-input";

                item.replaceChild(input, title);
                input.focus();

                input.onkeydown = (event) => {
                    if (event.key === "Enter") {
                        session.title =
                            input.value.trim() || "Без названия";
                        saveSessions();
                        renderHistory();
                    }

                    if (event.key === "Escape") {
                        renderHistory();
                    }
                };

                input.onblur = () => {
                    renderHistory();
                };
            };

            const deleteBtn = document.createElement("span");
            deleteBtn.className = "delete-btn";
            deleteBtn.innerText = "✕";

            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                deleteSession(id);
            };

            item.appendChild(title);
            item.appendChild(deleteBtn);

            item.onclick = () => loadSession(id);

            history.appendChild(item);
        });
}

function deleteSession(id) {

    delete sessions[id];

    if (id === sessionId) {
        sessionId = null;
        localStorage.removeItem("session_id");
        showNewChatWelcome();
    }

    saveSessions();
    renderHistory();
}

function saveSessions() {
    localStorage.setItem("sessions",
        JSON.stringify(sessions));
}





