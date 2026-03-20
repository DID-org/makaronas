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
    section: 'welcome',   // which section is visible
    error: null            // { message: string } | null
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
  // Event Binding
  // --------------------------------------------------------------------------

  function init() {
    // Welcome: start button (placeholder — Phase 3c wires session creation)
    var startBtn = document.getElementById('btn-start');
    if (startBtn) {
      startBtn.addEventListener('click', function () {
        // Placeholder: log to console, actual session creation is Phase 3c
        console.log('[Makaronas] Start button clicked — session creation wired in Phase 3c');
      });
    }

    // Error: retry button — return to welcome, clear error
    var retryBtn = document.getElementById('btn-retry');
    if (retryBtn) {
      retryBtn.addEventListener('click', function () {
        updateState({ section: 'welcome', error: null });
      });
    }

    // Keyboard: Escape dismisses error section, returns to welcome
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && state.section === 'error') {
        updateState({ section: 'welcome', error: null });
      }
    });

    // Initial render
    render();
  }

  // --------------------------------------------------------------------------
  // Public API (cross-module interface)
  // --------------------------------------------------------------------------

  window.App = {
    updateState: updateState,
    getState: getState,
    render: render
  };

  // --------------------------------------------------------------------------
  // Bootstrap
  // --------------------------------------------------------------------------

  document.addEventListener('DOMContentLoaded', init);

})();
