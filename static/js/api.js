/* ==========================================================================
   Makaronas — api.js
   API client: fetch wrapper, auth headers, error handling, path registry.
   All backend communication goes through window.Api.
   ========================================================================== */

(function () {
  'use strict';

  // --------------------------------------------------------------------------
  // Path Registry — single source of truth for all API URLs (Vision §6.2.4)
  // --------------------------------------------------------------------------

  var PATHS = {
    session:       '/api/v1/student/session',
    sessionById:   function (id) { return '/api/v1/student/session/' + id; },
    current:       function (id) { return '/api/v1/student/session/' + id + '/current'; },
    next:          function (id) { return '/api/v1/student/session/' + id + '/next'; },
    choice:        function (id) { return '/api/v1/student/session/' + id + '/choice'; },
    respond:       function (id) { return '/api/v1/student/session/' + id + '/respond'; },
    debrief:       function (id) { return '/api/v1/student/session/' + id + '/debrief'; },
    generate:      function (id) { return '/api/v1/student/session/' + id + '/generate'; },
    radar:         function (id) { return '/api/v1/student/profile/' + id + '/radar'; },
    deleteProfile: function (id) { return '/api/v1/student/profile/' + id; },
    exportProfile: function (id) { return '/api/v1/student/profile/' + id + '/export'; },
    asset:         function (taskId, src) { return '/api/v1/assets/' + taskId + '/' + src; }
  };

  // --------------------------------------------------------------------------
  // Error Code Map — backend code → I18n key
  // --------------------------------------------------------------------------

  var ERROR_MAP = {
    'SESSION_NOT_FOUND':  'error_session_expired',
    'TASK_NOT_FOUND':     'error_task_not_found',
    'NO_TASK_ASSIGNED':   'error_no_task',
    'NO_ACTIVE_PHASE':    'error_no_phase',
    'AI_UNAVAILABLE':     'error_ai_unavailable',
    'AI_TIMEOUT':         'error_ai_timeout',
    'UNAUTHORIZED':       'error_unauthorized',
    'FORBIDDEN':          'error_unauthorized',
    'RATE_LIMITED':       'error_rate_limit'
  };

  // --------------------------------------------------------------------------
  // ApiError Constructor
  // --------------------------------------------------------------------------

  /**
   * Structured error thrown by all Api functions.
   * code:       backend uppercase error code (e.g. 'SESSION_NOT_FOUND')
   * message:    Lithuanian translation from I18n
   * statusCode: HTTP status (useful for retry logic on 429)
   * data:       optional response data (e.g. TASK_CONTENT_UPDATED carries initial_phase)
   */
  function ApiError(code, message, statusCode, data) {
    this.name = 'ApiError';
    this.code = code || 'UNKNOWN';
    this.message = message || '';
    this.statusCode = statusCode || 0;
    this.data = data || null;
  }
  ApiError.prototype = Object.create(Error.prototype);
  ApiError.prototype.constructor = ApiError;

  // --------------------------------------------------------------------------
  // Helpers
  // --------------------------------------------------------------------------

  /**
   * Translates a backend error code to a Lithuanian message via I18n.
   * Falls back to I18n.error_generic, then a \u-escaped Lithuanian fallback.
   */
  function translateError(code) {
    var key = ERROR_MAP[code];
    if (key && window.I18n && window.I18n[key]) {
      return window.I18n[key];
    }
    // Unknown code — generic fallback
    return (window.I18n && window.I18n.error_generic) || 'Klaida';
  }

  // --------------------------------------------------------------------------
  // Core Request Function
  // --------------------------------------------------------------------------

  /**
   * Central fetch wrapper. All public Api functions delegate here.
   *
   * method:  HTTP method string ('GET', 'POST', 'DELETE')
   * url:     full path (from PATHS registry)
   * body:    request body (object, will be JSON-stringified) or null
   * options: { skipLock: bool, authToken: string }
   *          skipLock  — skip lockout guard (for background checks like getCurrentSession)
   *          authToken — explicit token override (for createSession before state has session)
   *
   * Returns unwrapped data on success. Throws ApiError on any failure.
   */
  function request(method, url, body, options) {
    var opts = options || {};
    var skipLock = opts.skipLock === true;
    var explicitToken = opts.authToken || null;

    var currentState = App.getState();

    // Lockout guard: prevent duplicate submissions (Vision §6.2.14)
    if (!skipLock && currentState.locked) {
      return Promise.reject(
        new ApiError('LOCKED', translateError('LOCKED'), 0)
      );
    }

    // Set locked state before the request
    if (!skipLock) {
      App.updateState({ locked: true });
    }

    // Build headers
    var headers = {
      'Accept': 'application/json'
    };
    if (body !== null && body !== undefined) {
      headers['Content-Type'] = 'application/json';
    }

    // Auth token: explicit override > state.session.auth_token
    var token = explicitToken;
    if (!token && currentState.session && currentState.session.auth_token) {
      token = currentState.session.auth_token;
    }
    if (token) {
      headers['Authorization'] = 'Bearer ' + token;
    }

    // Build fetch options
    var fetchOpts = {
      method: method,
      headers: headers
    };
    if (body !== null && body !== undefined) {
      fetchOpts.body = JSON.stringify(body);
    }

    // Execute fetch and handle all error paths
    return fetch(url, fetchOpts)
      .then(function (response) {
        var statusCode = response.status;

        // Try to parse JSON body (works for both success and structured errors)
        return response.json()
          .then(function (json) {
            // Case 1: Valid ApiResponse envelope
            if (json && typeof json.ok === 'boolean') {
              if (json.ok) {
                return json.data;
              }
              // Error envelope — extract code and translate
              var code = (json.error && json.error.code) || 'UNKNOWN';
              var message = translateError(code);
              throw new ApiError(code, message, statusCode, json.data || null);
            }
            // Case 2: JSON but not an ApiResponse envelope (unexpected)
            if (statusCode >= 400) {
              throw new ApiError('HTTP_ERROR', translateError('HTTP_ERROR'), statusCode);
            }
            // 2xx with non-envelope JSON — return as-is (defensive)
            return json;
          })
          .catch(function (err) {
            // Re-throw ApiError instances from the inner .then
            if (err instanceof ApiError) {
              throw err;
            }
            // JSON parse failed — non-JSON response body
            if (statusCode === 401 || statusCode === 403) {
              throw new ApiError('UNAUTHORIZED', translateError('UNAUTHORIZED'), statusCode);
            }
            throw new ApiError('HTTP_ERROR', translateError('HTTP_ERROR'), statusCode);
          });
      })
      .catch(function (err) {
        // Re-throw ApiError (already structured)
        if (err instanceof ApiError) {
          throw err;
        }
        // Network failure (fetch itself threw — offline, DNS, CORS)
        var networkMsg = (window.I18n && window.I18n.error_network) || 'Tinklo klaida';
        throw new ApiError('NETWORK_ERROR', networkMsg, 0);
      })
      .then(
        function (data) {
          // Success path: unlock
          if (!skipLock) {
            App.updateState({ locked: false });
          }
          return data;
        },
        function (err) {
          // Error path: unlock, then re-throw
          if (!skipLock) {
            App.updateState({ locked: false });
          }
          throw err;
        }
      );
  }

  // --------------------------------------------------------------------------
  // Public API Functions
  // --------------------------------------------------------------------------

  /**
   * Creates a new session. Requires an auth token since no session exists yet.
   * The backend's FakeAuthService accepts any non-empty Bearer token.
   * Phase 3c will generate a UUID and pass it here.
   */
  function createSession(authToken) {
    return request('POST', PATHS.session, {}, { authToken: authToken });
  }

  /**
   * Recovers current session state (read-only). Used on page reload.
   * Skips lockout — this is a background check, not a user action (§7.7).
   */
  function getCurrentSession(sessionId) {
    return request('GET', PATHS.current(sessionId), null, { skipLock: true });
  }

  /**
   * Loads a task into the session. Optional taskId for specific task.
   */
  function loadTask(sessionId, taskId) {
    var url = PATHS.next(sessionId);
    if (taskId) {
      url += '?task_id=' + encodeURIComponent(taskId);
    }
    return request('GET', url, null);
  }

  /**
   * Submits a button choice / investigation submit — advances the phase.
   */
  function submitChoice(sessionId, targetPhase, contextLabel) {
    var body = { target_phase: targetPhase };
    if (contextLabel !== null && contextLabel !== undefined) {
      body.context_label = contextLabel;
    }
    return request('POST', PATHS.choice(sessionId), body);
  }

  /**
   * Triggers content generation (empathy flip). POST /generate.
   */
  function generate(sessionId, sourceContent, studentPrompt) {
    return request('POST', PATHS.generate(sessionId), {
      source_content: sourceContent,
      student_prompt: studentPrompt
    });
  }

  /**
   * Deletes a student profile (GDPR right to deletion).
   */
  function deleteProfile(studentId) {
    return request('DELETE', PATHS.deleteProfile(studentId), null);
  }

  /**
   * Exports a student profile (GDPR right to access).
   */
  function exportProfile(studentId) {
    return request('GET', PATHS.exportProfile(studentId), null);
  }

  /**
   * Gets the radar data for a student.
   */
  function getRadar(studentId) {
    return request('GET', PATHS.radar(studentId), null);
  }

  // --------------------------------------------------------------------------
  // Asset URL Helper (not a fetch — returns URL string for img.src, etc.)
  // --------------------------------------------------------------------------

  function assetUrl(taskId, src) {
    return PATHS.asset(taskId, src);
  }

  // --------------------------------------------------------------------------
  // Public API (cross-module interface)
  // --------------------------------------------------------------------------

  window.Api = {
    // Session lifecycle
    createSession: createSession,
    getCurrentSession: getCurrentSession,
    loadTask: loadTask,
    submitChoice: submitChoice,
    generate: generate,

    // GDPR
    deleteProfile: deleteProfile,
    exportProfile: exportProfile,

    // Radar
    getRadar: getRadar,

    // Utility
    assetUrl: assetUrl,

    // Exposed for Phase 3b SSE functions to reuse
    PATHS: PATHS
  };

})();
