const AUTH_KEY = "mindbridge.auth";

const state = {
  sessionId: null,
  sending: false,
  loadingConversation: false,
  conversationRequest: 0,
  profile: null,
  modelName: "mock"
};

const els = {
  serviceState: document.querySelector("#serviceState"),
  modelState: document.querySelector("#modelState"),
  activeAccount: document.querySelector("#activeAccount"),
  switchAccount: document.querySelector("#switchAccount"),
  messages: document.querySelector("#messages"),
  historyList: document.querySelector("#historyList"),
  historyState: document.querySelector("#historyState"),
  refreshHistory: document.querySelector("#refreshHistory"),
  chatForm: document.querySelector("#chatForm"),
  messageInput: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
  newSession: document.querySelector("#newSession"),
  sessionBadge: document.querySelector("#sessionBadge"),
  studentGreeting: document.querySelector("#studentGreeting"),
  charCount: document.querySelector("#charCount")
};

function readAuth() {
  try {
    return JSON.parse(sessionStorage.getItem(AUTH_KEY) || "null");
  } catch {
    return null;
  }
}

function clearAuth() {
  sessionStorage.removeItem(AUTH_KEY);
}

function authHeader() {
  const auth = readAuth();
  if (!auth?.token) {
    window.location.replace("/");
    return "";
  }
  return `Basic ${auth.token}`;
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}), Authorization: authHeader() };
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response;
}

function setPill(el, text, tone = "ok") {
  el.textContent = text;
  el.className = `pill ${tone}`;
}

function isAdmin(profile) {
  return profile.roles?.some((role) => role.authority === "ROLE_ADMIN");
}

function displayModel(model) {
  return (model || "").includes("mindbridge-qwen2.5-7b-ft") ? "微调 Qwen2.5-7B" : model;
}

function studentDisplayName(profile) {
  return profile.displayName === "Demo Student" ? "Student" : (profile.displayName || profile.username || "Student");
}

function greetingForNow() {
  const hour = new Date().getHours();
  if (hour < 6) return "夜深了";
  if (hour < 11) return "早上好";
  if (hour < 14) return "中午好";
  if (hour < 18) return "下午好";
  return "晚上好";
}

function updateCharCount() {
  els.charCount.textContent = `${els.messageInput.value.length} / 1000`;
}

async function checkHealth() {
  try {
    const response = await fetch("/actuator/health");
    const body = await response.json();
    setPill(els.serviceState, body.status === "UP" ? "服务正常" : `服务 ${body.status}`, body.status === "UP" ? "ok" : "danger");
  } catch {
    setPill(els.serviceState, "服务 DOWN", "danger");
  }
}

async function loadProfile() {
  try {
    const response = await api("/api/profile");
    const profile = await response.json();
    if (isAdmin(profile)) {
      window.location.replace("/admin.html");
      return null;
    }
    state.profile = profile;
    const displayName = studentDisplayName(profile);
    els.activeAccount.textContent = displayName;
    els.studentGreeting.textContent = `${greetingForNow()}，${displayName}`;
    return profile;
  } catch {
    clearAuth();
    window.location.replace("/");
    return null;
  }
}

async function loadAgentStatus() {
  const response = await api("/api/agent/status");
  const status = await response.json();
  state.modelName = status.model || "mock";
  if (status.realModelEnabled) {
    setPill(els.modelState, `${status.provider} / ${displayModel(state.modelName)}`, "ok");
  } else {
    setPill(els.modelState, "mock 演示", "warn");
  }
}

function clearWelcome() {
  const empty = els.messages.querySelector(".empty");
  if (empty) empty.remove();
}

function addMessage(role, content) {
  clearWelcome();
  const row = document.createElement("article");
  row.className = `message ${role}`;
  row.innerHTML = `
    <div class="message-role">${role === "user" ? "我" : "MindBridge"}</div>
    <div class="bubble"></div>
  `;
  row.querySelector(".bubble").textContent = content;
  els.messages.append(row);
  els.messages.scrollTop = els.messages.scrollHeight;
  return row.querySelector(".bubble");
}

function formatSessionTime(raw) {
  if (!raw) return "";
  const normalized = /(?:Z|[+-]\d\d:\d\d)$/.test(raw) ? raw : `${raw}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}

function updateHistorySelection() {
  els.historyList.querySelectorAll(".history-item").forEach((button) => {
    const selected = button.dataset.sessionId === state.sessionId;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-current", selected ? "true" : "false");
  });
}

function renderHistory(sessions) {
  els.historyList.innerHTML = "";
  if (!sessions.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.innerHTML = "<strong>还没有历史对话</strong><span>发送第一条消息后，会话会保存在这里。</span>";
    els.historyList.append(empty);
    els.historyState.textContent = "从此刻开始记录";
    return;
  }

  sessions.forEach((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    button.dataset.sessionId = session.sessionId;

    const title = document.createElement("strong");
    title.textContent = session.title || "未命名对话";
    const meta = document.createElement("span");
    const messageText = `${session.messageCount || 0} 条消息`;
    const timeText = formatSessionTime(session.updatedAt);
    meta.textContent = timeText ? `${messageText} · ${timeText}` : messageText;

    button.append(title, meta);
    button.addEventListener("click", () => loadConversation(session.sessionId));
    els.historyList.append(button);
  });

  els.historyState.textContent = `${sessions.length} 个会话，点击即可继续`;
  updateHistorySelection();
}

async function loadSessions() {
  els.historyList.setAttribute("aria-busy", "true");
  els.refreshHistory.disabled = true;
  els.historyState.textContent = "正在读取...";
  try {
    const response = await api("/api/chat/sessions");
    renderHistory(await response.json());
  } catch (error) {
    els.historyState.textContent = "读取失败，请点击刷新重试";
    if (!els.historyList.children.length) {
      const failure = document.createElement("div");
      failure.className = "history-empty error";
      const title = document.createElement("strong");
      title.textContent = "暂时无法读取";
      const detail = document.createElement("span");
      detail.textContent = error.message;
      failure.append(title, detail);
      els.historyList.append(failure);
    }
  } finally {
    els.historyList.setAttribute("aria-busy", "false");
    els.refreshHistory.disabled = false;
  }
}

async function loadConversation(sessionId) {
  if (state.sending || state.loadingConversation || sessionId === state.sessionId) return;
  const requestId = ++state.conversationRequest;
  state.loadingConversation = true;
  els.sendButton.disabled = true;
  els.newSession.disabled = true;
  els.historyState.textContent = "正在打开对话...";
  setPill(els.sessionBadge, "LOADING", "warn");

  try {
    const response = await api(`/api/chat/sessions/${encodeURIComponent(sessionId)}`);
    const conversation = await response.json();
    if (requestId !== state.conversationRequest) return;

    state.sessionId = conversation.sessionId;
    els.messages.innerHTML = "";
    if (conversation.messages?.length) {
      conversation.messages.forEach((message) => addMessage(message.role.toLowerCase(), message.content));
    } else {
      els.messages.innerHTML = `
        <div class="empty student-welcome">
          <strong>这个会话还没有消息</strong>
          <p>你可以在下面继续写下想说的话。</p>
        </div>
      `;
    }
    els.historyState.textContent = conversation.title || "已打开历史对话";
    updateHistorySelection();
    setPill(els.sessionBadge, "CONTINUE", "ok");
    els.messageInput.focus();
  } catch (error) {
    els.historyState.textContent = `打开失败：${error.message}`;
    setPill(els.sessionBadge, "ERROR", "danger");
  } finally {
    if (requestId === state.conversationRequest) {
      state.loadingConversation = false;
      els.sendButton.disabled = false;
      els.newSession.disabled = false;
    }
  }
}

function parseSse(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const rest = parts.pop();
  for (const part of parts) {
    const dataLine = part.split("\n").find((line) => line.startsWith("data: "));
    if (!dataLine) continue;
    onEvent(JSON.parse(dataLine.slice(6)));
  }
  return rest;
}

async function sendMessage(event) {
  event.preventDefault();
  if (state.sending || state.loadingConversation) return;
  const message = els.messageInput.value.trim();
  if (!message) return;
  state.sending = true;
  els.sendButton.disabled = true;
  setPill(els.sessionBadge, "THINKING", "warn");
  els.messageInput.value = "";
  updateCharCount();
  addMessage("user", message);
  const assistant = addMessage("assistant", "");
  let raw = "";

  try {
    const response = await api("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId: state.sessionId, message })
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let streamFailed = false;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSse(buffer, (eventData) => {
        if (eventData.type === "meta") {
          state.sessionId = eventData.sessionId;
          updateHistorySelection();
        }
        if (eventData.type === "token") {
          raw += eventData.content || "";
          assistant.textContent = raw;
          els.messages.scrollTop = els.messages.scrollHeight;
        }
        if (eventData.type === "error") {
          streamFailed = true;
          if (!raw) assistant.textContent = eventData.message || "MCP 工具调用失败";
          setPill(els.sessionBadge, "ERROR", "danger");
        }
      });
    }
    if (!streamFailed) setPill(els.sessionBadge, "DONE", "ok");
  } catch (error) {
    assistant.textContent = `发送失败：${error.message}`;
    setPill(els.sessionBadge, "ERROR", "danger");
  } finally {
    state.sending = false;
    els.sendButton.disabled = false;
    loadSessions();
  }
}

function resetSession() {
  state.conversationRequest += 1;
  state.loadingConversation = false;
  state.sessionId = null;
  els.messages.innerHTML = `
    <div class="empty student-welcome">
      <span class="welcome-kicker">A FRESH START</span>
      <strong>新会话已经准备好</strong>
      <p>不用延续刚才的话题，你可以从此刻最想说的一件事重新开始。</p>
    </div>
  `;
  els.messageInput.value = "";
  updateCharCount();
  updateHistorySelection();
  els.sendButton.disabled = false;
  els.newSession.disabled = false;
  els.historyState.textContent = "选择一条记录可继续以前的对话";
  setPill(els.sessionBadge, "READY");
}

function logout() {
  clearAuth();
  window.location.assign("/");
}

document.querySelectorAll("[data-quick]").forEach((button) => {
  button.addEventListener("click", () => {
    els.messageInput.value = button.dataset.quick;
    updateCharCount();
    els.messageInput.focus();
  });
});
els.messageInput.addEventListener("input", updateCharCount);
els.chatForm.addEventListener("submit", sendMessage);
els.newSession.addEventListener("click", resetSession);
els.refreshHistory.addEventListener("click", loadSessions);
els.switchAccount.addEventListener("click", logout);

checkHealth();
loadProfile().then((profile) => {
  if (profile) Promise.all([loadAgentStatus(), loadSessions()]);
});
updateCharCount();
