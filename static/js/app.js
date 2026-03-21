/* ==========================================================================
   Makaronas — app.js
   Entry point: central state, updateState/render cycle, section switching.
   ========================================================================== */

(function () {
  'use strict';

  // --------------------------------------------------------------------------
  // State
  // --------------------------------------------------------------------------

  var state = {
    section: 'welcome',    // which section is visible
    error: null,           // { message: string } | null
    locked: false,         // UI lockout during API calls (Phase 3a)
    session: null,         // { session_id: string, auth_token: string } | null
    task: null,            // Full phase data dict from API (_derive_phase_response)
    phase: null,           // Current phase ID string (shorthand for task.current_phase)
    taskSequenceIndex: 0,  // Position in TASK_SEQUENCE array
    terminal: null         // { evaluation_outcome, reveal } when is_terminal, null otherwise
  };

  // --------------------------------------------------------------------------
  // Task Sequence (honest stub — replaced by V9 Roadmap Engine)
  // --------------------------------------------------------------------------

  var TASK_SEQUENCE = ['task-01', 'task-04'];

  // --------------------------------------------------------------------------
  // Session Storage Keys (namespaced to avoid collisions)
  // --------------------------------------------------------------------------

  var STORAGE_KEYS = {
    sessionId: 'makaronas_session_id',
    authToken: 'makaronas_auth_token',
    interactionState: 'makaronas_interaction_state'
  };

  // --------------------------------------------------------------------------
  // State Management
  // --------------------------------------------------------------------------

  /**
   * Merges changes into state and triggers a render.
   * All state mutations go through this function.
   */
  function updateState(changes) {
    Object.assign(state, changes);
    // Derived state: error implies error section
    if (state.error && state.section !== 'error') {
      state.section = 'error';
    }
    render();
  }

  /**
   * Returns a shallow copy of state — prevents external mutation.
   */
  function getState() {
    return Object.assign({}, state);
  }

  // --------------------------------------------------------------------------
  // Render
  // --------------------------------------------------------------------------

  /**
   * Master render — reads state and updates DOM.
   * Future phases add conditional render calls here.
   */
  function render() {
    renderSection(state.section);
    renderError(state.error);
  }

  /**
   * Sets the active section via data attribute.
   * CSS handles show/hide — JS only sets the attribute.
   */
  function renderSection(sectionId) {
    var app = document.getElementById('app');
    if (app) {
      app.dataset.activeSection = sectionId;
    }
    // Scroll to top when switching sections
    window.scrollTo(0, 0);

    // Focus management: move focus to the new section's heading.
    // setTimeout(0) ensures the display change has painted before focusing —
    // without this, .focus() silently fails on elements transitioning from
    // display: none in some browsers (especially Safari).
    setTimeout(function () {
      var section = document.getElementById('section-' + sectionId);
      if (section) {
        var heading = section.querySelector('h2[tabindex="-1"]');
        if (heading) {
          heading.focus();
        }
      }
    }, 0);
  }

  /**
   * Updates the error section content.
   * Pure render — reads error state, updates DOM text only.
   */
  function renderError(error) {
    var messageEl = document.getElementById('error-message');
    if (!messageEl) return;

    if (error && error.message) {
      messageEl.textContent = error.message;
    } else if (error) {
      // Error object without message — use Lithuanian generic fallback
      messageEl.textContent = (window.I18n && window.I18n.error_generic) || '';
    } else {
      messageEl.textContent = '';
    }
  }

  // --------------------------------------------------------------------------
  // Phase Rendering + Task Loading
  // --------------------------------------------------------------------------

  /**
   * Renders a phase into the content and interaction panels.
   * Called on initial task load, phase transitions, and session recovery.
   * This is imperative — not called from the render() loop.
   */
  function renderPhase(phaseData) {
    // Update state with new phase data
    updateState({
      task: phaseData,
      phase: phaseData.current_phase,
      section: 'task'
    });

    // Clear dialogue state from previous phase (Phase 5b)
    if (window.Dialogue) {
      window.Dialogue.clearDialogue();
    }

    // Content panel: preserve heading, render presentation blocks
    var contentPanel = document.querySelector('.content-panel');
    if (contentPanel && window.Renderer) {
      var contentHeading = contentPanel.querySelector('h2');
      window.Renderer.renderBlocksInto(contentPanel, phaseData.content || [], phaseData.task_id);
      if (contentHeading) {
        contentHeading.textContent = phaseData.title || ((window.I18n && window.I18n.content_heading) || 'Turinys');
        contentPanel.insertBefore(contentHeading, contentPanel.firstChild);
      }
    }

    // Layout class: reset and apply variant based on interaction type
    var taskLayout = document.querySelector('.task-layout');
    if (taskLayout) {
      taskLayout.className = 'task-layout';
      if (phaseData.interaction && phaseData.interaction.type === 'investigation') {
        taskLayout.classList.add('layout-investigation');
      }
    }

    // Interaction panel: render controls via interactions.js
    if (window.Interactions) {
      window.Interactions.renderInteraction(phaseData);
    }

    // Focus management: move focus to content heading for screen readers
    setTimeout(function () {
      var heading = contentPanel && contentPanel.querySelector('h2[tabindex="-1"]');
      if (heading) {
        heading.focus();
      }
    }, 0);
  }

  /**
   * Loads the first task from TASK_SEQUENCE after session creation.
   * Chains from createSessionFlow().
   */
  function loadFirstTask() {
    if (!state.session) return;

    var taskId = TASK_SEQUENCE[state.taskSequenceIndex];
    if (!taskId) {
      updateState({ error: { message: (window.I18n && window.I18n.task_load_error) || 'Task load error' } });
      return;
    }

    Api.loadTask(state.session.session_id, taskId).then(function (data) {
      renderPhase(data);
      console.log('[Makaronas] Task loaded:', taskId);
    }).catch(function (err) {
      updateState({ error: { message: err.message || (window.I18n && window.I18n.task_load_error) || 'Task load error' } });
    });
  }

  /**
   * Unified entry point for all phase transitions.
   * Both button clicks (5a) and SSE done events (5b) route here.
   */
  function handlePhaseTransition(phaseData) {
    if (phaseData.is_terminal) {
      // Store terminal data for Phase 6b debrief/reveal
      updateState({
        terminal: {
          evaluation_outcome: phaseData.evaluation_outcome,
          reveal: phaseData.reveal
        }
      });
    } else {
      // Clear stale terminal data from previous phases
      if (state.terminal !== null) {
        updateState({ terminal: null });
      }
    }
    // Always render the phase content (even terminal — shows final content)
    renderPhase(phaseData);
  }

  // --------------------------------------------------------------------------
  // Session Helpers
  // --------------------------------------------------------------------------

  /**
   * Clears all session data from sessionStorage and state.
   * Safe to call even when no session exists (no-op for missing keys).
   */
  function clearSession() {
    sessionStorage.removeItem(STORAGE_KEYS.sessionId);
    sessionStorage.removeItem(STORAGE_KEYS.authToken);
    sessionStorage.removeItem(STORAGE_KEYS.interactionState);
    updateState({ session: null });
  }

  /**
   * Creates a new backend session and persists credentials.
   * Called by the start button click handler.
   */
  function createSessionFlow() {
    if (state.locked) return;

    var authToken = crypto.randomUUID();

    Api.createSession(authToken).then(function (data) {
      sessionStorage.setItem(STORAGE_KEYS.sessionId, data.session_id);
      sessionStorage.setItem(STORAGE_KEYS.authToken, authToken);
      updateState({ session: { session_id: data.session_id, auth_token: authToken } });
      console.log('[Makaronas] Session created:', data.session_id);
      loadFirstTask();
    }).catch(function (err) {
      updateState({ error: { message: err.message } });
    });
  }

  /**
   * Attempts to recover an existing session from sessionStorage.
   * Called once during init() — if credentials exist, queries the backend
   * for current session state and restores it.
   */
  function recoverSession() {
    var sessionId;
    var authToken;

    try {
      sessionId = sessionStorage.getItem(STORAGE_KEYS.sessionId);
      authToken = sessionStorage.getItem(STORAGE_KEYS.authToken);
    } catch (e) {
      // sessionStorage unavailable (rare: old Safari private browsing, quota)
      // Fall through to welcome screen
      return;
    }

    if (!sessionId || !authToken) return;

    // Populate state BEFORE API call — getCurrentSession reads
    // state.session.auth_token for the Bearer header (see IMPL_NOTES 3a)
    updateState({ session: { session_id: sessionId, auth_token: authToken } });

    Api.getCurrentSession(sessionId).then(function (data) {
      if (data.current_task === null) {
        // Session alive but no active task — show welcome
        console.log('[Makaronas] Recovery: session alive, no active task');
        return;
      }
      // Active task — render the current phase
      state.dialogueHistory = data.dialogue_history || [];
      renderPhase(data);

      // Restore dialogue history if the current phase is a freeform interaction (Phase 5b)
      if (state.dialogueHistory.length > 0 && window.Dialogue) {
        var recoveryPanel = document.querySelector('.interaction-panel');
        if (recoveryPanel) {
          window.Dialogue.renderDialogueHistory(recoveryPanel, state.dialogueHistory);
        }
      }

      console.log('[Makaronas] Recovery: restored active task');
    }).catch(function (err) {
      if (err.code === 'SESSION_NOT_FOUND' || err.code === 'TASK_CONTENT_UPDATED') {
        // Session expired or task changed — clean start
        clearSession();
        console.log('[Makaronas] Recovery:', err.code, '— cleared session');
        return;
      }
      // Network or unexpected error — clear session, show error
      clearSession();
      updateState({ error: { message: err.message } });
    });
  }

  // --------------------------------------------------------------------------
  // Event Binding
  // --------------------------------------------------------------------------

  function init() {
    // Welcome: start button — creates backend session
    var startBtn = document.getElementById('btn-start');
    if (startBtn) {
      startBtn.addEventListener('click', createSessionFlow);
    }

    // Error: retry button — return to welcome, clear session + error
    var retryBtn = document.getElementById('btn-retry');
    if (retryBtn) {
      retryBtn.addEventListener('click', function () {
        clearSession();
        updateState({ section: 'welcome', error: null });
      });
    }

    // Keyboard: Escape dismisses error section, returns to welcome
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && state.section === 'error') {
        clearSession();
        updateState({ section: 'welcome', error: null });
      }
    });

    // Attempt session recovery from sessionStorage before first render
    recoverSession();

    // Initial render (recoverSession may have already triggered renders
    // via updateState, but this ensures welcome is shown if no recovery)
    render();
  }

  // --------------------------------------------------------------------------
  // Public API (cross-module interface)
  // --------------------------------------------------------------------------

  window.App = {
    updateState: updateState,
    getState: getState,
    render: render,
    renderPhase: renderPhase,
    loadFirstTask: loadFirstTask,
    handlePhaseTransition: handlePhaseTransition
  };

  // --------------------------------------------------------------------------
  // Bootstrap
  // --------------------------------------------------------------------------

  document.addEventListener('DOMContentLoaded', init);

})();
