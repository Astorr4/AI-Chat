// =========================
// MARKED CONFIG
// =========================

marked.setOptions({
    breaks: true,
    gfm: true
});

function sanitizeHtml(html) {
    if (window.DOMPurify) {
        return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
    }

    // Безопасный fallback: показываем контент как plain text,
    // если санитайзер недоступен.
    const temp = document.createElement("div");
    temp.textContent = html;
    return temp.innerHTML;
}

let sessionId = localStorage.getItem("session_id");
let sessions = JSON.parse(localStorage.getItem("sessions") || "{}");
let pendingRequests = JSON.parse(localStorage.getItem("pending_requests") || "{}");
let analyzeFileEnabled = false;
let currentAnalysisFileName = null;

const STREAM_RENDER_INTERVAL_MS = 80;
const STREAM_SAVE_INTERVAL_MS = 500;

function createThrottle(fn, wait) {
    let lastTime = 0;
    let timer = null;

    const throttled = (...args) => {
        const now = Date.now();
        const remaining = wait - (now - lastTime);

        if (remaining <= 0) {
            if (timer) {
                clearTimeout(timer);
                timer = null;
            }
            lastTime = now;
            fn(...args);
            return;
        }

        if (!timer) {
            timer = setTimeout(() => {
                timer = null;
                lastTime = Date.now();
                fn(...args);
            }, remaining);
        }
    };

    throttled.flush = (...args) => {
        if (timer) {
            clearTimeout(timer);
            timer = null;
        }
        lastTime = Date.now();
        fn(...args);
    };

    return throttled;
}

function savePendingRequests() {
    localStorage.setItem("pending_requests", JSON.stringify(pendingRequests));
}

function setFileStatus(text, isEnabled) {
    const fileStatus = document.getElementById("fileStatus");
    const analyzeBtn = document.getElementById("analyzeFileBtn");

    if (fileStatus) {
        fileStatus.textContent = text;
        fileStatus.classList.toggle("active", !!isEnabled);
    }

    if (analyzeBtn) {
        analyzeBtn.classList.toggle("active", !!isEnabled);
    }
}

function updateSessionAnalysisState(enabled, fileName) {
    if (!sessionId || !sessions[sessionId]) return;
    sessions[sessionId].analysis = {
        enabled: !!enabled,
        fileName: fileName || null,
    };
    saveSessions();
}

async function uploadFileForAnalysis(file) {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const headers = {};
    if (sessionId) {
        headers["X-Session-Id"] = sessionId;
    }

    const response = await fetch("/file-analysis/upload", {
        method: "POST",
        headers,
        body: formData,
    });

    if (!response.ok) {
        let errorMessage = `HTTP ${response.status}`;
        try {
            const payload = await response.json();
            if (payload && payload.error) {
                errorMessage = payload.error;
            }
        } catch (_) {}
        throw new Error(errorMessage);
    }

    const payload = await response.json();

    const serverSessionId = response.headers.get("X-Session-Id")
        || payload.session_id
        || sessionId
        || ("session_" + Date.now());

    if (!sessions[serverSessionId]) {
        sessions[serverSessionId] = {
            title: `Анализ: ${payload.file_name || file.name}`.substring(0, 40),
            messages: [],
            created: Date.now(),
            analysis: {
                enabled: true,
                fileName: payload.file_name || file.name,
            },
        };
    }

    sessionId = serverSessionId;
    localStorage.setItem("session_id", sessionId);

    analyzeFileEnabled = true;
    currentAnalysisFileName = payload.file_name || file.name;

    updateSessionAnalysisState(true, currentAnalysisFileName);
    renderHistory();

    const truncatedSuffix = payload.truncated
        ? " (контекст укорочен)"
        : "";
    setFileStatus(
        `Режим: анализ файла (${currentAnalysisFileName})${truncatedSuffix}`,
        true
    );
}

function setupFileAnalysisUI() {
    const analyzeBtn = document.getElementById("analyzeFileBtn");
    const fileInput = document.getElementById("fileInput");

    if (!analyzeBtn || !fileInput) return;

    analyzeBtn.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", async (e) => {
        const selectedFile = e.target.files && e.target.files[0]
            ? e.target.files[0]
            : null;

        if (!selectedFile) return;

        try {
            setFileStatus("Загрузка файла для анализа...", false);
            await uploadFileForAnalysis(selectedFile);
        } catch (err) {
            analyzeFileEnabled = false;
            currentAnalysisFileName = null;
            updateSessionAnalysisState(false, null);
            setFileStatus(`Ошибка анализа файла: ${String(err.message || err)}`, false);
        } finally {
            fileInput.value = "";
        }
    });

    if (sessionId && sessions[sessionId] && sessions[sessionId].analysis?.enabled) {
        analyzeFileEnabled = true;
        currentAnalysisFileName = sessions[sessionId].analysis.fileName || null;
        if (currentAnalysisFileName) {
            setFileStatus(`Режим: анализ файла (${currentAnalysisFileName})`, true);
        } else {
            setFileStatus("Режим: анализ файла", true);
        }
    } else {
        setFileStatus("Режим: общий чат", false);
    }
}

function registerPendingRequest(requestId, payload) {
    pendingRequests[requestId] = payload;
    savePendingRequests();
}

function clearPendingRequest(requestId) {
    if (pendingRequests[requestId]) {
        delete pendingRequests[requestId];
        savePendingRequests();
    }
}

function getOrCreateAssistantContainer(targetSessionId, assistantIndex) {
    const chat = document.getElementById("chat-box");
    if (!chat) return null;

    const existingNode = chat.querySelector(`.message.assistant[data-assistant-index="${assistantIndex}"]`);
    if (existingNode) {
        return {
            assistantMessage: existingNode,
            contentDiv: existingNode.querySelector(".assistant-content"),
        };
    }

    const assistantMessage = appendMessage("assistant", "");
    if (!assistantMessage) return null;

    assistantMessage.dataset.assistantIndex = String(assistantIndex);
    const contentDiv = assistantMessage.querySelector(".assistant-content");
    if (contentDiv) {
        const typingIndicator = document.createElement("div");
        typingIndicator.className = "typing";
        typingIndicator.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        contentDiv.appendChild(typingIndicator);
    }

    return { assistantMessage, contentDiv };
}

function applyAssistantMeta(contentDiv, sources, confidence) {
    if (!contentDiv) return;

    if (confidence !== null && confidence < 0.5) {
        const warning = document.createElement("div");
        warning.className = "low-confidence-warning";
        warning.innerHTML = `
            ⚠ Ответ может быть неточным.
            Попробуйте уточнить формулировку запроса.
        `;
        contentDiv.appendChild(warning);
    }

    if (sources.length > 0 || confidence !== null) {
        const metaBlock = document.createElement("div");
        metaBlock.className = "sources";

        let html = "";

        if (sources.length > 0) {
            html += "<strong>Источники:</strong><br>";
            html += sources.map(s => `• ${s}`).join("<br>");
        }

        if (confidence !== null && confidence !== undefined) {
            const percent = Math.round(confidence * 100);
            let levelClass = "conf-low";
            if (percent >= 80) levelClass = "conf-high";
            else if (percent >= 60) levelClass = "conf-medium";

            html += `
                <div class="confidence ${levelClass}">
                    Уверенность: ${percent}%
                </div>
            `;
        }

        metaBlock.innerHTML = html;
        contentDiv.appendChild(metaBlock);
    }
}

async function consumeAssistantStream({
    reader,
    currentSessionId,
    assistantIndex,
    contentDiv,
    assistantMessage,
}) {
    const decoder = new TextDecoder();

    let fullText = "";
    let contentText = "";
    let sources = [];
    let confidence = null;

    const renderAssistantContentThrottled = createThrottle(() => {
        const typing = contentDiv.querySelector(".typing");
        if (typing) typing.remove();

        if (contentText.trim() === "В документации информация не найдена.") {
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
            const parsedHtml = marked.parse(contentText);
            const safeHtml = sanitizeHtml(parsedHtml);
            contentDiv.innerHTML = safeHtml;

            contentDiv
                .querySelectorAll("pre code")
                .forEach(block => {
                    hljs.highlightElement(block);
                });
        }
    }, STREAM_RENDER_INTERVAL_MS);

    const persistAssistantMessageThrottled = createThrottle(() => {
        const msg = sessions[currentSessionId]?.messages?.[assistantIndex];
        if (!msg) return;

        msg.content = contentText;
        msg.html = sanitizeHtml(contentDiv.innerHTML);
        saveSessions();
    }, STREAM_SAVE_INTERVAL_MS);

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        fullText += decoder.decode(value);

        const confSplit = fullText.split("###CONFIDENCE###");
        const confidencePart = confSplit.length > 1
            ? confSplit.slice(1).join("###CONFIDENCE###")
            : "";
        const mainPart = confSplit[0];

        if (confidencePart) {
            const parsedConfidence = parseFloat(confidencePart.trim());
            if (!Number.isNaN(parsedConfidence)) {
                confidence = parsedConfidence;
            }
        }

        const sourceSplit = mainPart.split("###SOURCES###");
        const answerWithMeta = sourceSplit[0];

        if (sourceSplit.length > 1) {
            try {
                sources = JSON.parse(sourceSplit[1].trim());
            } catch (e) {}
        }

        contentText = answerWithMeta;

        renderAssistantContentThrottled();
        persistAssistantMessageThrottled();
    }

    renderAssistantContentThrottled.flush();
    persistAssistantMessageThrottled.flush();

    const msg = sessions[currentSessionId]?.messages?.[assistantIndex];
    if (msg) {
        msg.sources = sources;
        msg.confidence = confidence;
    }

    applyAssistantMeta(contentDiv, sources, confidence);
    
    // Обновляем панель источников
    if (typeof sources === "object" && !Array.isArray(sources)) {
        renderSourcesPanel(sources);
    } else if (Array.isArray(sources) && sources.length > 0) {
        // Обратная совместимость со старой структурой
        const legacySources = {};
        sources.forEach(s => {
            legacySources[s] = [{ text: "Текст недоступен", score: 0 }];
        });
        renderSourcesPanel(legacySources);
    }

    if (msg) {
        msg.html = sanitizeHtml(contentDiv.innerHTML);
        saveSessions();
    }

    addCopyButton(assistantMessage);
}

async function resumePendingStreamsForCurrentSession() {
    if (!sessionId || !sessions[sessionId]) return;

    const entries = Object.entries(pendingRequests)
        .filter(([, p]) => p && p.sessionId === sessionId);

    for (const [requestId, payload] of entries) {
        try {
            const ui = getOrCreateAssistantContainer(sessionId, payload.assistantIndex);
            if (!ui || !ui.contentDiv) {
                clearPendingRequest(requestId);
                continue;
            }

            // Check if this is a file analysis request by checking the session state
            const isFileAnalysisMode = !!(
                sessions[sessionId]?.analysis?.enabled
            );

            const endpoint = isFileAnalysisMode 
                ? `/file-analysis-stream/${encodeURIComponent(requestId)}`
                : `/chat-stream/${encodeURIComponent(requestId)}`;

            const response = await fetch(endpoint, {
                headers: {
                    "X-Session-Id": sessionId,
                },
            });
            if (!response.ok || !response.body) {
                clearPendingRequest(requestId);
                continue;
            }

            await consumeAssistantStream({
                reader: response.body.getReader(),
                currentSessionId: sessionId,
                assistantIndex: payload.assistantIndex,
                contentDiv: ui.contentDiv,
                assistantMessage: ui.assistantMessage,
            });

            clearPendingRequest(requestId);
        } catch (_) {
            // Оставляем pending, чтобы можно было повторить после следующего reload
        }
    }
}

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

    resumePendingStreamsForCurrentSession();
    setupFileAnalysisUI();
    
    // Инициализация панели источников
    initSourcesPanel();
    updateSourcesFromLastMessage();

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
    analyzeFileEnabled = false;
    currentAnalysisFileName = null;
    localStorage.removeItem("session_id");
    setFileStatus("Режим: общий чат", false);
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
            created: Date.now(),
            analysis: {
                enabled: false,
                fileName: null,
            },
        };
        localStorage.setItem("session_id", sessionId);
    }

    const isFileAnalysisMode = !!(
        analyzeFileEnabled
        && sessions[sessionId]
        && sessions[sessionId].analysis
        && sessions[sessionId].analysis.enabled
    );

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
    const assistantIndex = sessions[sessionId].messages.length - 1;
    if (assistantMessage) {
        assistantMessage.dataset.assistantIndex = String(assistantIndex);
    }
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

        const response = await fetch(isFileAnalysisMode ? "/file-analysis/chat" : "/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-Session-Id": sessionId
            },
            body: JSON.stringify({ question: message })
        });

        if (!response.ok) {
            let errorMessage = `HTTP ${response.status}`;
            try {
                const payload = await response.json();
                if (payload && payload.error) {
                    errorMessage = payload.error;
                }
            } catch (_) {}

            throw new Error(errorMessage);
        }

        if (!response.body) {
            throw new Error("Пустой ответ сервера");
        }

        const requestId = response.headers.get("X-Request-Id");
        if (requestId) {
            registerPendingRequest(requestId, {
                sessionId,
                assistantIndex,
                createdAt: Date.now(),
                isFileAnalysis: isFileAnalysisMode
            });
        }

        await consumeAssistantStream({
            reader: response.body.getReader(),
            currentSessionId: sessionId,
            assistantIndex,
            contentDiv,
            assistantMessage,
        });

        if (requestId) {
            clearPendingRequest(requestId);
        }

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

        const typing = contentDiv.querySelector(".typing");
        if (typing) typing.remove();

        contentDiv.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">⚠</div>
                <div class="no-results-title">Ошибка запроса</div>
                <div class="no-results-text">${sanitizeHtml(String(err.message || err))}</div>
            </div>
        `;

        let lastMsg =
            sessions[sessionId].messages[
                sessions[sessionId].messages.length - 1
            ];
        lastMsg.content = "";
        lastMsg.html = sanitizeHtml(contentDiv.innerHTML);
        saveSessions();
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
        contentDiv.innerHTML = sanitizeHtml(text);
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

    const analysis = sessions[id]?.analysis;
    analyzeFileEnabled = !!analysis?.enabled;
    currentAnalysisFileName = analysis?.fileName || null;
    if (analyzeFileEnabled) {
        if (currentAnalysisFileName) {
            setFileStatus(`Режим: анализ файла (${currentAnalysisFileName})`, true);
        } else {
            setFileStatus("Режим: анализ файла", true);
        }
    } else {
        setFileStatus("Режим: общий чат", false);
    }

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
                    contentDiv.innerHTML = sanitizeHtml(msg.html);
                } else {
                    const parsedHtml = marked.parse(msg.content || "");
                    contentDiv.innerHTML = sanitizeHtml(parsedHtml);
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
    
    // Обновляем панель источников
    updateSourcesFromLastMessage();
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
        analyzeFileEnabled = false;
        currentAnalysisFileName = null;
        localStorage.removeItem("session_id");
        setFileStatus("Режим: общий чат", false);
        showNewChatWelcome();
    }

    saveSessions();
    renderHistory();
}

function saveSessions() {
    localStorage.setItem("sessions",
        JSON.stringify(sessions));
}

// =========================
// SOURCES PANEL
// =========================

let currentSources = {};

function initSourcesPanel() {
    const toggleBtn = document.getElementById("sourcesToggle");
    const panel = document.getElementById("sourcesPanel");
    
    if (toggleBtn && panel) {
        toggleBtn.addEventListener("click", () => {
            panel.classList.toggle("collapsed");
        });
    }
}

function renderSourcesPanel(sources) {
    currentSources = sources || {};
    const sourcesList = document.getElementById("sourcesList");
    const sourcesCount = document.getElementById("sourcesCount");
    
    if (!sourcesList || !sourcesCount) return;
    
    const sourceNames = Object.keys(sources);
    sourcesCount.textContent = sourceNames.length;
    
    if (sourceNames.length === 0) {
        sourcesList.innerHTML = `
            <div class="source-empty">
                <div class="source-empty-icon">📄</div>
                <div>Нет источников</div>
            </div>
        `;
        return;
    }
    
    let html = "";
    sourceNames.forEach((sourceName, index) => {
        const chunks = sources[sourceName] || [];
        const chunksHtml = chunks.map(chunk => {
            const score = chunk.score ? Math.round(chunk.score * 100) : 0;
            const text = chunk.text.length > 300 
                ? chunk.text.substring(0, 300) + "..." 
                : chunk.text;
            return `
                <div class="source-chunk">
                    <div class="source-chunk-score">Релевантность: ${score}%</div>
                    ${escapeHtml(text)}
                </div>
            `;
        }).join("");
        
        html += `
            <div class="source-item${index === 0 ? ' expanded' : ''}" data-source="${escapeHtml(sourceName)}">
                <div class="source-header" onclick="toggleSource(this)">
                    <span class="source-name">${escapeHtml(sourceName)}</span>
                    <span class="source-expand">
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                    </span>
                </div>
                <div class="source-chunks">
                    ${chunksHtml}
                </div>
            </div>
        `;
    });
    
    sourcesList.innerHTML = html;
}

function toggleSource(headerEl) {
    const sourceItem = headerEl.closest(".source-item");
    if (sourceItem) {
        sourceItem.classList.toggle("expanded");
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function updateSourcesFromLastMessage() {
    if (!sessionId || !sessions[sessionId]) return;
    
    const messages = sessions[sessionId].messages || [];
    let lastSources = {};
    
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg.role === "assistant" && msg.sources) {
            // Проверяем новую структуру (объект с текстами чанков)
            if (typeof msg.sources === "object" && !Array.isArray(msg.sources)) {
                lastSources = msg.sources;
            }
            break;
        }
    }
    
    renderSourcesPanel(lastSources);
}





