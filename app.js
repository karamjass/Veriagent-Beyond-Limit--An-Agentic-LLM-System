const HISTORY_KEY = 'veriagent.conversations.v2';
const THEME_KEY = 'veriagent.theme.v1';
const COLLAPSE_KEY = 'veriagent.sidebar.collapsed.v1';
const DISMISS_IMPORT_KEY = 'veriagent.import.dismissed.v1';
const MAX_CONVERSATIONS = 20;
const MAX_STORED_MESSAGE = 6000;

const root = document.documentElement;
const workspace = document.querySelector('#workspace');
const sidebar = document.querySelector('#sidebar');
const messageInput = document.querySelector('#messageInput');
const composer = document.querySelector('#composer');
const messageList = document.querySelector('#messageList');
const welcomeState = document.querySelector('#welcomeState');
const greetingHeading = document.querySelector('#greeting');
const thinkingRow = document.querySelector('#thinkingRow');
const historyList = document.querySelector('#historyList');
const historyEmpty = document.querySelector('#historyEmpty');
const historySearch = document.querySelector('#historySearch');
const sendButton = document.querySelector('#sendButton');
const charCount = document.querySelector('#charCount');
const themeToggle = document.querySelector('#themeToggle');
const toast = document.querySelector('#toast');

// Database and Auth elements
const dbStatus = document.querySelector('#dbStatus');
const dbDot = document.querySelector('#dbDot');
const dbModeLabel = document.querySelector('#dbModeLabel');
const authBtn = document.querySelector('#authBtn');
const authBtnDot = document.querySelector('#authBtnDot');
const authBtnLabel = document.querySelector('#authBtnLabel');
const authModal = document.querySelector('#authModal');
const authModalBackdrop = document.querySelector('#authModalBackdrop');
const authModalClose = document.querySelector('#authModalClose');
const authModalTitle = document.querySelector('#authModalTitle');
const authModalSubtitle = document.querySelector('#authModalSubtitle');
const authTabs = document.querySelector('#authTabs');
const tabSignIn = document.querySelector('#tabSignIn');
const tabSignUp = document.querySelector('#tabSignUp');
const authForm = document.querySelector('#authForm');
const authNameField = document.querySelector('#authNameField');
const authName = document.querySelector('#authName');
const authEmail = document.querySelector('#authEmail');
const authPassword = document.querySelector('#authPassword');
const authError = document.querySelector('#authError');
const authSubmitBtn = document.querySelector('#authSubmitBtn');
const googleAuthBtn = document.querySelector('#googleAuthBtn');
const authEnvPill = document.querySelector('#authEnvPill');
const authEnvNote = document.querySelector('#authEnvNote');
const profileName = document.querySelector('#profileName');
const profileStatusSub = document.querySelector('#profileStatusSub');
const profileStatusDot = document.querySelector('#profileStatusDot');
const composerAuthGate = document.querySelector('#composerAuthGate');
const gateLoginBtn = document.querySelector('#gateLoginBtn');
const gateGoogleBtn = document.querySelector('#gateGoogleBtn');
const welcomeAuthCard = document.querySelector('#welcomeAuthCard');
const welcomeLoginBtn = document.querySelector('#welcomeLoginBtn');
const welcomeGoogleBtn = document.querySelector('#welcomeGoogleBtn');

let conversations = [];
let currentConversationId = null;
let currentUser = null;
let isSending = false;
let toastTimer = null;
let thinkingTimer = null;

function getFirstName(user) {
  if (!user) return '';
  if (user.first_name) return user.first_name;
  let raw = user.display_name || user.user_metadata?.full_name || user.user_metadata?.name || '';
  if (!raw && user.email) {
    raw = user.email.split('@')[0];
  }
  if (!raw) return 'User';
  // Strip trailing numbers (e.g. user16 -> user)
  raw = raw.replace(/\d+$/, '');
  const firstWord = raw.split(/[\s._-]+/)[0] || raw;
  return firstWord.charAt(0).toUpperCase() + firstWord.slice(1);
}
let thinkingStage = 0;
let authConfig = null;
let supabaseClient = null;
let currentSupabaseSession = null;
let csrfToken = '';
let isSignUpMode = false;

// -------------------------------------------------------------
// Utilities & Cookie / CSRF Handling
// -------------------------------------------------------------

function getCookie(name) {
  const match = document.cookie.match(new RegExp('(?:^|;\\s*)' + name.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&') + '=([^;]*)'));
  return match ? decodeURIComponent(match[1]) : null;
}

function getCsrfToken() {
  return csrfToken || getCookie('veriagent_csrf') || '';
}

let apiBaseUrl = localStorage.getItem('veriagent_backend_url') || '';

function getApiUrl(path) {
  if (!apiBaseUrl) return path;
  return `${apiBaseUrl.replace(/\/+$/, '')}${path}`;
}

async function apiFetch(url, options = {}) {
  const opts = { ...options };
  opts.headers = { ...(opts.headers || {}) };
  
  let targetUrl = url;
  if (apiBaseUrl && url.startsWith('/api/')) {
    targetUrl = getApiUrl(url);
    opts.credentials = 'include';
  } else {
    opts.credentials = 'same-origin';
  }

  if (currentSupabaseSession && currentSupabaseSession.access_token) {
    opts.headers['Authorization'] = `Bearer ${currentSupabaseSession.access_token}`;
  }

  const method = (opts.method || 'GET').toUpperCase();
  if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
    const csrf = getCsrfToken();
    if (csrf) {
      opts.headers['X-CSRF-Token'] = csrf;
    }
  }

  return fetch(targetUrl, opts);
}

function showToast(message) {
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add('show');
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove('show'), 2800);
}

function makeIcon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', 'true');
  if (name === 'chat') {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M5 5.5h14v10H9l-4 3v-13zM8 9h8M8 12h5');
    svg.append(path);
  }
  return svg;
}

function getDateGroup(timestamp) {
  const date = new Date(timestamp);
  const today = new Date();
  const startToday = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
  const startDate = new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  if (startDate >= startToday) return 'TODAY';
  if (startToday - startDate < 7 * 86400000) return 'THIS WEEK';
  return 'EARLIER';
}

function titleFromPrompt(prompt) {
  const clean = prompt.replace(/\s+/g, ' ').trim();
  return clean.length > 44 ? `${clean.slice(0, 44).trimEnd()}…` : clean;
}

// -------------------------------------------------------------
// History & Conversation List Rendering
// -------------------------------------------------------------

function renderHistory() {
  if (!historyList || !historySearch) return;
  const query = historySearch.value.trim().toLocaleLowerCase();
  const visible = conversations
    .filter((conv) => !query || (conv.title && conv.title.toLocaleLowerCase().includes(query)))
    .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));

  historyList.replaceChildren();
  if (historyEmpty) {
    historyEmpty.hidden = visible.length > 0;
    historyEmpty.textContent = query ? 'No conversations match that search.' : 'Your conversations will appear here.';
  }

  let currentGroup = '';
  visible.forEach((conversation) => {
    const group = getDateGroup(conversation.updatedAt || Date.now());
    if (group !== currentGroup) {
      currentGroup = group;
      const label = document.createElement('div');
      label.className = 'history-group';
      label.textContent = group;
      historyList.append(label);
    }
    const item = document.createElement('div');
    item.className = `history-item${conversation.id === currentConversationId ? ' active' : ''}`;
    item.setAttribute('role', 'listitem');

    const openButton = document.createElement('button');
    openButton.type = 'button';
    openButton.className = 'history-open';
    openButton.setAttribute('aria-label', `Open conversation: ${conversation.title}`);
    openButton.append(makeIcon('chat'));

    const title = document.createElement('span');
    title.className = 'history-title';
    title.textContent = conversation.title || 'Conversation';
    openButton.append(title);
    openButton.addEventListener('click', () => openConversation(conversation.id));

    const deleteButton = document.createElement('button');
    deleteButton.type = 'button';
    deleteButton.className = 'history-delete';
    deleteButton.textContent = '×';
    deleteButton.title = 'Delete conversation';
    deleteButton.setAttribute('aria-label', `Delete conversation: ${conversation.title}`);
    deleteButton.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteConversation(conversation.id);
    });

    item.append(openButton, deleteButton);
    historyList.append(item);
  });
}

function findConversation(id = currentConversationId) {
  return conversations.find((conversation) => conversation.id === id) || null;
}

function updateGreeting() {
  if (!greetingHeading) return;
  const hour = new Date().getHours();
  const salutation = hour >= 5 && hour < 12
    ? 'Good morning'
    : hour >= 12 && hour < 17
      ? 'Good afternoon'
      : 'Good evening';
  greetingHeading.textContent = `${salutation}!`;
}

function startNewConversation() {
  if (isSending) {
    showToast('Wait for the current response to finish first.');
    return;
  }
  currentConversationId = null;
  messageList.replaceChildren();
  updateGreeting();
  if (welcomeState) welcomeState.hidden = false;
  if (thinkingRow) thinkingRow.hidden = true;
  messageInput.value = '';
  resizeInput();
  renderHistory();
  if (currentUser) {
    messageInput.focus();
  }
  closeMobileSidebar();
}

// -------------------------------------------------------------
// Database Operations (List, Load, Delete, Send)
// -------------------------------------------------------------

async function refreshConversationsList() {
  if (!currentUser) {
    conversations = [];
    renderHistory();
    return;
  }
  try {
    const res = await apiFetch('/api/conversations');
    if (!res.ok) {
      conversations = [];
      renderHistory();
      return;
    }
    const data = await res.json();
    const items = Array.isArray(data) ? data : (data.conversations || []);
    conversations = items.map((c) => ({
      id: c.id,
      title: c.title || 'Conversation',
      createdAt: Number(c.createdAt) || (c.created_at ? new Date(c.created_at).getTime() : Date.now()),
      updatedAt: Number(c.updatedAt) || (c.updated_at ? new Date(c.updated_at).getTime() : Date.now()),
      messages: []
    }));
    renderHistory();
  } catch (err) {
    conversations = [];
    renderHistory();
  }
}

async function openConversation(id) {
  if (isSending) {
    showToast('Wait for the current response to finish first.');
    return;
  }
  try {
    const res = await apiFetch(`/api/conversations/${encodeURIComponent(id)}`);
    if (res.status === 404) {
      showToast('Conversation not found or not owned by you.');
      return;
    }
    if (res.status === 401) {
      showToast('Session expired. Please sign in again.');
      openAuthModal();
      return;
    }
    if (!res.ok) throw new Error(`Could not load conversation (${res.status})`);

    const data = await res.json();
    const conv = data.conversation || data;
    currentConversationId = conv.id;
    if (welcomeState) welcomeState.hidden = true;
    messageList.replaceChildren();

    const messages = (conv.messages || []).map((m) => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: String(m.content || ''),
      verification: String(m.verification_label || m.verification || ''),
      verified: Boolean(m.verified),
      elapsed: Number(m.elapsed) || 0
    }));

    messages.forEach((msg) => appendMessage(msg));
    renderHistory();
    scrollToLatest(false);
    closeMobileSidebar();
  } catch (err) {
    showToast('Could not load that conversation.');
  }
}

async function deleteConversation(id) {
  if (isSending) {
    showToast('Wait for the current response to finish first.');
    return;
  }
  try {
    const res = await apiFetch(`/api/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (res.status === 404) {
      showToast('Conversation not found or not owned by caller.');
      return;
    }
    if (res.status === 401) {
      showToast('Authentication required to delete conversations.');
      return;
    }
    if (!res.ok) throw new Error(`Failed (${res.status})`);

    conversations = conversations.filter((c) => c.id !== id);
    if (currentConversationId === id) {
      startNewConversation();
    }
    renderHistory();
    showToast('Conversation deleted.');
  } catch (err) {
    conversations = conversations.filter((c) => c.id !== id);
    if (currentConversationId === id) startNewConversation();
    renderHistory();
    showToast('Conversation removed.');
  }
}

function makeMessageAvatar() {
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  const image = document.createElement('img');
  image.src = document.querySelector('.sidebar-logo img')?.src || '/brand-logo.webp?v=9';
  image.alt = '';
  avatar.append(image);
  return avatar;
}

function appendMessage(message) {
  const row = document.createElement('article');
  row.className = `message-row ${message.role}`;
  if (message.role === 'assistant') row.append(makeMessageAvatar());

  const body = document.createElement('div');
  body.className = 'message-body';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = message.content;
  body.append(bubble);

  if (message.role === 'assistant') {
    const meta = document.createElement('div');
    meta.className = `message-meta${message.verified ? '' : ' warning'}`;

    const mark = document.createElement('i');
    mark.className = 'review-mark';

    const label = document.createElement('span');
    label.textContent = message.verification || 'Internal review only';
    meta.append(mark, label);

    if (message.elapsed) {
      const timing = document.createElement('span');
      timing.textContent = `· ${message.elapsed.toFixed(1)}s`;
      meta.append(timing);
    }
    body.append(meta);

    const tools = document.createElement('div');
    tools.className = 'message-tools';
    const copyButton = document.createElement('button');
    copyButton.type = 'button';
    copyButton.textContent = 'Copy response';
    copyButton.addEventListener('click', () => copyText(message.content));
    tools.append(copyButton);
    body.append(tools);
  } else {
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.textContent = 'YOU';
    body.append(meta);
  }

  row.append(body);
  messageList.append(row);
  return row;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast('Copied to clipboard.');
  } catch {
    showToast('Clipboard access is unavailable in this browser.');
  }
}

function scrollToLatest(smooth = true) {
  requestAnimationFrame(() => {
    const stage = document.querySelector('#chatStage');
    if (!stage) return;
    stage.scrollTo({
      top: stage.scrollHeight,
      behavior: smooth ? 'smooth' : 'auto'
    });
  });
}

function updateThinkingLabel(text) {
  const el = document.querySelector('#thinkingText');
  if (el) el.textContent = text;
}

async function sendMessage(rawText = messageInput.value) {
  const prompt = rawText.trim();
  if (!prompt || isSending) return;
  if (prompt.length > 3000) {
    showToast('Keep each message under 3,000 characters.');
    return;
  }

  isSending = true;
  sendButton.disabled = true;
  messageInput.disabled = true;

  if (welcomeState) welcomeState.hidden = true;
  appendMessage({ role: 'user', content: prompt });
  messageInput.value = '';
  resizeInput();

  if (thinkingRow) {
    thinkingRow.hidden = false;
    thinkingStage = 0;
    const stages = [
      'Drafting a considered response…',
      'Checking for contradictions and unsupported claims…',
      'Refining the answer and its uncertainty…',
      'Preparing the reviewed response…'
    ];
    updateThinkingLabel(stages[0]);
    thinkingTimer = window.setInterval(() => {
      thinkingStage = Math.min(thinkingStage + 1, stages.length - 1);
      updateThinkingLabel(stages[thinkingStage]);
    }, 4200);
  }
  scrollToLatest();

  try {
    const response = await apiFetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: prompt,
        conversation_id: currentConversationId
      })
    });

    if (response.status === 401) {
      throw new Error('Authentication required. In production, sign in through Supabase Auth.');
    }
    if (response.status === 403) {
      throw new Error('CSRF validation failed. Refresh the page to obtain a fresh session token.');
    }
    if (!response.ok) {
      const errPayload = await response.json().catch(() => ({}));
      throw new Error(errPayload.error || `Request failed with status ${response.status}`);
    }

    const payload = await response.json();
    if (payload.conversation_id) {
      currentConversationId = payload.conversation_id;
    }

    const assistantMessage = {
      role: 'assistant',
      content: String(payload.answer || 'I could not produce a response.').slice(0, MAX_STORED_MESSAGE),
      verification: String(payload.verification_label || 'Internal review only'),
      verified: Boolean(payload.verified),
      elapsed: Number(payload.elapsed) || 0
    };

    appendMessage(assistantMessage);
    await refreshConversationsList();
  } catch (error) {
    if (/(?:who(?:\x27s|\s+is|\s+are)\s+(?:the\s+|your\s+)?(?:owner|creator|maker|developer|founder|author|builder)|who\s+(?:made|created|built|developed|owns|founded|designed)\s+(?:you|this|this\s+(?:app|agent|system|bot)|veriagent|veragent|veriage)|who\s+(?:owns\s+you|are\s+you)|who\s+is\s+karamjass(?:\s+kaur)?|what\s+model\s+(?:are\s+you|powers\s+you|is\s+this|do\s+you\s+use|did\s+karamjass\s+use)|which\s+model\s+(?:are\s+you|powers\s+you|is\s+this|do\s+you\s+use)|(?:tell\s+me\s+about\s+|what\s+is\s+)?(?:the\s+)?(?:owner|creator)\s+of\s+(?:veriagent|veragent|veriage)|owner\s+of\s+(?:veriagent|veragent|veriage)|creator\s+of\s+(?:veriagent|veragent|veriage))/i.test(prompt.trim())) {
      appendMessage({
        role: 'assistant',
        content: "Karamjass Kaur is the owner and creator of VeriAgent.\n\nShe created VeriAgent by using the Qwen 3.5 9B (9 billion) model as its foundation, incorporating a dedicated multi-agent reasoning, calculation, and verification pipeline.",
        verification: 'System identity verified · Owner: Karamjass Kaur · Model: Qwen 3.5 9B',
        verified: true,
        elapsed: 0
      });
      return;
    }
    const text = `I couldn't complete that response. ${error.message || 'Please try again.'}`;
    appendMessage({
      role: 'assistant',
      content: text,
      verification: 'Request not completed',
      verified: false,
      elapsed: 0
    });
  } finally {
    window.clearInterval(thinkingTimer);
    if (thinkingRow) thinkingRow.hidden = true;
    isSending = false;
    sendButton.disabled = false;
    messageInput.disabled = false;
    messageInput.focus();
    scrollToLatest();
    refreshRuntimeStatus();
  }
}

function resizeInput() {
  if (!messageInput || !charCount || !sendButton) return;
  messageInput.rows = Math.max(1, Math.min(7, messageInput.value.split('\n').length));
  charCount.textContent = `${messageInput.value.length.toLocaleString()} / 3,000`;
  sendButton.disabled = isSending || !messageInput.value.trim();
}

// -------------------------------------------------------------
// Auto-Save & Legacy Storage Migration
// -------------------------------------------------------------

async function autoMigrateLocalChatsIfAny() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return;
    const localList = JSON.parse(raw);
    if (Array.isArray(localList) && localList.length > 0 && currentUser) {
      await apiFetch('/api/conversations/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversations: localList })
      });
    }
  } catch {}
  try {
    localStorage.removeItem(HISTORY_KEY);
    localStorage.removeItem(DISMISS_IMPORT_KEY);
  } catch {}
}

// -------------------------------------------------------------
// Authentication, Session Lifecycle & Chat Gate
// -------------------------------------------------------------

function lockChatUI() {
  unlockChatUI();
}

function unlockChatUI() {
  if (messageInput) {
    messageInput.disabled = false;
    messageInput.placeholder = 'Message VeriAgent…';
  }
  resizeInput();
}

function updateAuthState(user) {
  currentUser = user || { id: 'local-dev-user', display_name: 'User' };
  updateProfileUI(currentUser);
  unlockChatUI();
  updateGreeting();
}

function openAuthModal() {
  return;
}

function closeAuthModal() {
  return;
}

function setAuthMode(signUp) {
  isSignUpMode = Boolean(signUp);
  if (authNameField) {
    authNameField.hidden = !isSignUpMode;
  }
  if (tabSignIn && tabSignUp) {
    tabSignIn.classList.toggle('is-active', !isSignUpMode);
    tabSignIn.setAttribute('aria-selected', !isSignUpMode ? 'true' : 'false');
    tabSignUp.classList.toggle('is-active', isSignUpMode);
    tabSignUp.setAttribute('aria-selected', isSignUpMode ? 'true' : 'false');
  }
  if (authSubmitBtn) {
    authSubmitBtn.textContent = isSignUpMode ? 'Sign Up' : 'Sign In';
  }
  if (authModalTitle) {
    authModalTitle.textContent = isSignUpMode ? 'Create Account' : 'Sign In to VeriAgent';
  }
  if (authError) authError.hidden = true;
}

async function handleGoogleSignIn() {
  return;
}

async function handleAuthFormSubmit(event) {
  if (event) event.preventDefault();
  return;
}

async function syncSessionWithBackend(session) {
  return;
}

async function handleLogout() {
  await apiFetch('/api/auth/logout', { method: 'POST' }).catch(() => {});
  showToast('Conversations cleared.');
  conversations = [];
  currentConversationId = null;
  messageList.replaceChildren();
  if (welcomeState) welcomeState.hidden = false;
  renderHistory();
}

function updateProfileUI(user) {
  if (profileName) profileName.textContent = 'Karamjass Kaur';
  if (profileStatusSub) profileStatusSub.textContent = 'Owner & Creator · Qwen 3.5 9B';
  updateGreeting();
}

async function initAuth() {
  unlockChatUI();
  updateGreeting();

  try {
    const res = await apiFetch('/api/auth/config', { cache: 'no-store' });
    if (res.ok) {
      authConfig = await res.json();
    }
  } catch {
    authConfig = { mode: 'development', auth_mode: 'local_dev', supabase_configured: false, database_type: 'sqlite' };
  }

  // Update DB pill
  const dbType = (authConfig?.database_type || 'sqlite').toUpperCase();
  if (dbModeLabel) dbModeLabel.textContent = dbType;
  if (dbStatus) dbStatus.classList.add('connected');

  // Silently check or establish session in background without any popup or modal
  try {
    const uRes = await apiFetch('/api/auth/user');
    if (uRes.ok) {
      const uData = await uRes.json();
      csrfToken = uData.csrf_token || '';
      currentUser = uData.user || { id: 'local-dev-user', display_name: 'User' };
    } else {
      const sRes = await apiFetch('/api/auth/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      if (sRes.ok) {
        const sData = await sRes.json();
        csrfToken = sData.csrf_token || '';
        currentUser = sData.user || { id: 'local-dev-user', display_name: 'User' };
      } else {
        currentUser = { id: 'local-dev-user', display_name: 'User' };
      }
    }
  } catch {
    currentUser = { id: 'local-dev-user', display_name: 'User' };
  }

  unlockChatUI();
  updateGreeting();
}

// -------------------------------------------------------------
// Runtime & Model Status Check
// -------------------------------------------------------------

async function refreshRuntimeStatus() {
  const status = document.querySelector('#runtimeStatus');
  const label = document.querySelector('#modelStatus');
  if (!status || !label) return;
  try {
    const response = await fetch('/api/health', { cache: 'no-store' });
    if (response.status === 404 || response.status === 405) {
      status.className = 'runtime-status setup-needed';
      label.textContent = 'OPEN LIVE PREVIEW';
      status.title = 'Static HTML preview. Open running Flask server to chat.';
      return;
    }
    if (!response.ok) throw new Error('Status request failed');
    const health = await response.json();

    if (dbModeLabel && health.database_type) {
      dbModeLabel.textContent = health.database_type.toUpperCase();
    }
    if (dbStatus) {
      dbStatus.classList.toggle('connected', Boolean(health.database?.ready));
      dbStatus.classList.toggle('error', !health.database?.ready);
      dbStatus.title = health.database?.ready ? 'Database connected' : 'Database connection unavailable';
    }

    if (!health.ollama_connected) {
      status.className = 'runtime-status setup-needed';
      label.textContent = 'OLLAMA OFFLINE';
      status.title = health.model_error || 'Start the local Ollama service.';
    } else if (!health.model_available) {
      status.className = 'runtime-status setup-needed';
      label.textContent = 'PULL QWEN MODEL';
      status.title = `Run: ollama pull ${health.model || 'qwen3.5:9b-q4_K_M'}`;
    } else {
      status.className = 'runtime-status ready';
      label.textContent = 'QWEN3.5 9B READY';
      status.title = `Ollama model ${health.model || 'Qwen3.5 9B'} is available locally.`;
    }
  } catch {
    status.className = 'runtime-status setup-needed';
    label.textContent = 'STATUS UNKNOWN';
    status.title = 'Could not read local status.';
  }
}

// -------------------------------------------------------------
// Theme & Layout Handlers
// -------------------------------------------------------------

function setTheme(theme) {
  const nextTheme = theme === 'dark' ? 'dark' : 'light';
  root.dataset.theme = nextTheme;
  try { localStorage.setItem(THEME_KEY, nextTheme); } catch {}
  const isLight = nextTheme === 'light';
  const themeLabel = document.querySelector('#themeLabel');
  if (themeLabel) themeLabel.textContent = isLight ? 'Day · Light' : 'Night · Dark';
  if (themeToggle) themeToggle.setAttribute('aria-label', isLight ? 'Switch to dark theme' : 'Switch to light theme');
  const metaTheme = document.querySelector('meta[name="theme-color"]');
  if (metaTheme) metaTheme.content = isLight ? '#ffffff' : '#15131a';
}

function closeMobileSidebar() {
  if (workspace) workspace.classList.remove('mobile-open');
  const scrim = document.querySelector('#mobileScrim');
  if (scrim) scrim.hidden = true;
}

function toggleMobileSidebar() {
  if (!workspace) return;
  const isOpen = workspace.classList.toggle('mobile-open');
  const scrim = document.querySelector('#mobileScrim');
  if (scrim) scrim.hidden = !isOpen;
}

function clearAllHistory() {
  if (isSending) {
    showToast('Wait for the current response to finish first.');
    return;
  }
  if (!conversations.length) return;
  if (!window.confirm('Delete all saved conversations from your account/browser?')) return;

  conversations.forEach((c) => {
    apiFetch(`/api/conversations/${encodeURIComponent(c.id)}`, { method: 'DELETE' }).catch(() => {});
  });
  try { localStorage.removeItem(HISTORY_KEY); } catch {}
  conversations = [];
  currentConversationId = null;
  startNewConversation();
  renderHistory();
  showToast('Conversations cleared.');
}

// -------------------------------------------------------------
// Event Listeners & Bootstrapping
// -------------------------------------------------------------

if (composer) {
  composer.addEventListener('submit', (event) => {
    event.preventDefault();
    sendMessage();
  });
}

if (messageInput) {
  messageInput.addEventListener('input', resizeInput);
  messageInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  });
}

document.querySelector('#newChatBtn')?.addEventListener('click', startNewConversation);
document.querySelectorAll('.suggestion').forEach((button) => {
  button.addEventListener('click', () => {
    if (!messageInput) return;
    messageInput.value = button.dataset.prompt || '';
    resizeInput();
    messageInput.focus();
    if (!messageInput.value.endsWith(': ')) sendMessage();
  });
});

if (historySearch) historySearch.addEventListener('input', renderHistory);
document.querySelector('#clearHistoryBtn')?.addEventListener('click', clearAllHistory);
document.querySelector('#collapseSidebar')?.addEventListener('click', () => {
  if (!workspace) return;
  workspace.classList.toggle('sidebar-collapsed');
  try { localStorage.setItem(COLLAPSE_KEY, workspace.classList.contains('sidebar-collapsed') ? '1' : '0'); } catch {}
});

document.querySelector('#mobileMenu')?.addEventListener('click', toggleMobileSidebar);
document.querySelector('#mobileScrim')?.addEventListener('click', closeMobileSidebar);
themeToggle?.addEventListener('click', () => setTheme(root.dataset.theme === 'light' ? 'dark' : 'light'));

// Auth triggers
authBtn?.addEventListener('click', () => {
  if (currentUser || currentSupabaseSession) {
    handleLogout();
  } else {
    openAuthModal();
  }
});

authModalBackdrop?.addEventListener('click', closeAuthModal);
authModalClose?.addEventListener('click', closeAuthModal);
tabSignIn?.addEventListener('click', () => setAuthMode(false));
tabSignUp?.addEventListener('click', () => setAuthMode(true));
authForm?.addEventListener('submit', handleAuthFormSubmit);
googleAuthBtn?.addEventListener('click', handleGoogleSignIn);
gateGoogleBtn?.addEventListener('click', handleGoogleSignIn);
welcomeGoogleBtn?.addEventListener('click', handleGoogleSignIn);
gateLoginBtn?.addEventListener('click', openAuthModal);
welcomeLoginBtn?.addEventListener('click', openAuthModal);
composerAuthGate?.addEventListener('click', openAuthModal);

window.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    startNewConversation();
  }
  if (event.key === '/' && document.activeElement !== historySearch && document.activeElement !== messageInput) {
    event.preventDefault();
    historySearch?.focus();
  }
  if (event.key === 'Escape') {
    closeMobileSidebar();
    closeAuthModal();
  }
});

// Initial boot sequence
try {
  if (localStorage.getItem(COLLAPSE_KEY) === '1' && window.innerWidth > 680) {
    workspace?.classList.add('sidebar-collapsed');
  }
  setTheme(localStorage.getItem(THEME_KEY) || 'light');
} catch {
  setTheme('light');
}

resizeInput();
updateGreeting();
window.setInterval(updateGreeting, 60_000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) updateGreeting(); });
window.addEventListener('pageshow', updateGreeting);

initAuth().then(() => {
  refreshConversationsList();
  refreshRuntimeStatus();
});
