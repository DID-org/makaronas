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
    report:        function (id) { return '/api/v1/student/session/' + id + '/report'; },
    radar:         function (id) { return '/api/v1/student/profile/' + id + '/radar'; },
    deleteProfile: function (id) { return '/api/v1/student/profile/' + id; },
    exportProfile: function (id) { return '/api/v1/student/profile/' + id + '/export'; },
    asset:         function (taskId, src) { return '/api/v1/assets/' + taskId + '/' + src; },
    clientError:   function (id) { return '/api/v1/student/session/' + id + '/client-error'; }
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

  function getReport(sessionId) {
    return request('GET', PATHS.report(sessionId), null);
  }

  function assetUrl(taskId, src) {
    return PATHS.asset(taskId, src);
  }

  /**
   * Reports a client-side error (asset load failure, AI stream error) to
   * session telemetry. Fire-and-forget — failures here are swallowed so
   * the student's flow is never affected. Requires a live session.
   */
  function reportClientError(errorType, details) {
    try {
      var appState = (window.App && window.App.getState && window.App.getState()) || {};
      var sessionId = appState.session && appState.session.session_id;
      if (!sessionId) return;
      request('POST', PATHS.clientError(sessionId), {
        error_type: errorType,
        details: details || {}
      }, { skipLock: true }).catch(function () {
        // Swallow — telemetry must never break the student's session.
      });
    } catch (e) {
      // Ditto — defensive.
    }
  }

  // --------------------------------------------------------------------------
  // SSE Parser — Manual ReadableStream parsing for streaming AI responses
  //
  // Why manual: EventSource only supports GET and can't send Authorization
  // headers. The backend's get_current_user() reads ONLY from the Authorization
  // header (no query param fallback). Both /respond (POST) and /debrief (GET)
  // need Bearer auth, so both use fetch + ReadableStream.
  //
  // Wire format (from backend/streaming.py format_sse_event):
  //   event: {type}\n
  //   data: {json}\n
  //   \n
  //
  // Four event types: token, done, error, redact.
  // Each event is exactly one "event:" line + one "data:" line + one blank line.
  // No multi-line data fields, no id/retry/comment lines.
  // --------------------------------------------------------------------------

  /**
   * Parses an SSE byte stream into typed callback dispatches.
   *
   * Handles three edge cases that make SSE parsing non-trivial:
   * 1. Partial chunks — an event split across network packets (buffered until
   *    the terminating \n\n arrives).
   * 2. Multi-event chunks — multiple complete events in one packet (each
   *    dispatched independently).
   * 3. Multi-byte UTF-8 splits — Lithuanian characters like š (2 bytes:
   *    0xC5 0xA1) can split across chunk boundaries. The TextDecoder with
   *    { stream: true } buffers incomplete byte sequences automatically.
   *
   * reader:    ReadableStream.getReader() result
   * decoder:   TextDecoder instance (must be created with { stream: true })
   * callbacks: { onToken, onDone, onRedact, onError }
   * onCleanup: called on stream end — handles unlock + caller cleanup
   */
  function parseSSEStream(reader, decoder, callbacks, onCleanup) {
    // Buffer accumulates text across chunks until a complete event
    // (terminated by \n\n) is found.
    var buffer = '';

    // Guard: once a terminal event fires (done/error/redact/parse-error),
    // stop dispatching and stop the pump. Without this, the pump loop
    // continues until the ReadableStream closes — which means callbacks
    // could fire AFTER a terminal callback if events arrive between the
    // terminal dispatch and the server closing the connection.
    var terminated = false;

    function terminate() {
      if (terminated) return;
      terminated = true;
      onCleanup();
    }

    function pump() {
      if (terminated) return;
      reader.read().then(function (result) {
        if (terminated) return;
        if (result.done) {
          // Stream closed by server. Any leftover buffer is an incomplete
          // event — discard it (the server always sends \n\n after each event,
          // so leftovers mean an abnormal close).
          terminate();
          return;
        }

        // Decode bytes to string. The { stream: true } option on TextDecoder
        // ensures that a multi-byte character split across chunks is buffered
        // internally and combined with the next chunk — no replacement chars.
        buffer += decoder.decode(result.value, { stream: true });

        // Split on double-newline — the SSE event terminator.
        // Each complete segment between \n\n boundaries is one event.
        var parts = buffer.split('\n\n');

        // The last element is either empty (buffer ended on \n\n) or an
        // incomplete event (no trailing \n\n yet). Keep it in the buffer.
        buffer = parts.pop();

        // Dispatch each complete event. Stop if a terminal event fires.
        for (var i = 0; i < parts.length; i++) {
          if (terminated) break;
          if (parts[i].trim() !== '') {
            dispatchSSEEvent(parts[i], callbacks, terminate);
          }
        }

        // Continue reading (unless a terminal event stopped us)
        if (!terminated) {
          pump();
        }
      }).catch(function (err) {
        if (terminated) return;
        // ReadableStream error — network disconnect, abort, etc.
        // AbortError is expected when the caller calls abort() — not a real error.
        if (err && err.name === 'AbortError') {
          terminate();
          return;
        }
        if (callbacks.onError) {
          var msg = (window.I18n && window.I18n.error_network) || 'Tinklo klaida';
          callbacks.onError('STREAM_ERROR', msg, '');
        }
        terminate();
      });
    }

    pump();
  }

  /**
   * Parses a single SSE event block and dispatches to the appropriate callback.
   *
   * A block looks like:
   *   event: token
   *   data: {"text": "chunk"}
   *
   * We extract the event type from the "event:" line and the JSON payload
   * from the "data:" line, then dispatch based on the type.
   */
  function dispatchSSEEvent(block, callbacks, onCleanup) {
    var lines = block.split('\n');
    var eventType = '';
    var dataStr = '';

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      // SSE spec: field name is everything before the first colon,
      // value is everything after the colon + optional leading space.
      if (line.indexOf('event:') === 0) {
        eventType = line.substring(6).trim();
      } else if (line.indexOf('data:') === 0) {
        dataStr = line.substring(5).trim();
      }
    }

    if (!eventType || !dataStr) {
      return; // Malformed event — skip silently
    }

    // Parse the JSON payload safely
    var data;
    try {
      data = JSON.parse(dataStr);
    } catch (e) {
      // Malformed JSON — surface as error, don't crash the parser
      if (callbacks.onError) {
        callbacks.onError(
          'PARSE_ERROR',
          translateError('PARSE_ERROR'),
          ''
        );
      }
      onCleanup();
      return;
    }

    // Dispatch based on event type.
    // "done", "error", and "redact" are terminal — after them the stream ends.
    switch (eventType) {
      case 'token':
        if (callbacks.onToken) {
          callbacks.onToken(data.text);
        }
        break;

      case 'done':
        if (callbacks.onDone) {
          callbacks.onDone(data.full_text, data.data);
        }
        onCleanup();
        break;

      case 'error':
        if (callbacks.onError) {
          callbacks.onError(data.code, data.message, data.partial_text || '');
        }
        onCleanup();
        break;

      case 'redact':
        // Redact replaces done — it's terminal. The backend emits redact
        // INSTEAD of done when a safety violation is detected (Framework P12).
        if (callbacks.onRedact) {
          callbacks.onRedact(data.fallback_text);
        }
        onCleanup();
        break;

      default:
        // Unknown event type — ignore. Forward compatibility.
        break;
    }
  }

  // --------------------------------------------------------------------------
  // SSE Streaming Functions
  // --------------------------------------------------------------------------

  /**
   * Opens a POST SSE connection to /respond for AI dialogue streaming.
   *
   * sessionId: active session ID
   * action:    interaction action string (e.g. "respond")
   * payload:   student's message text
   * callbacks: { onToken(text), onDone(fullText, data), onRedact(fallbackText),
   *              onError(code, message, partialText) }
   *
   * Returns { abort } — call abort() to cancel the stream and unlock the UI.
   */
  function streamRespond(sessionId, action, payload, callbacks) {
    var cbs = callbacks || {};
    var aborted = false;
    var cleanedUp = false;
    var controller = new AbortController();

    // Lock the UI for the duration of the stream (not just the fetch).
    // Differs from request() where lockout wraps a single fetch.
    App.updateState({ locked: true });

    // Read auth token from state (Phase 3c populates session before calling us)
    var state = App.getState();
    var token = (state.session && state.session.auth_token) || '';

    var headers = {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream'
    };
    if (token) {
      headers['Authorization'] = 'Bearer ' + token;
    }

    // Cleanup runs exactly once — on done, error, redact, or abort.
    function cleanup() {
      if (cleanedUp) return;
      cleanedUp = true;
      App.updateState({ locked: false });
    }

    fetch(PATHS.respond(sessionId), {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({ action: action, payload: payload }),
      signal: controller.signal
    }).then(function (response) {
      // Check for HTTP errors before entering the stream.
      // Once SSE starts, HTTP status is already 200 — errors come as events.
      if (!response.ok) {
        return response.json().then(function (json) {
          var code = (json.error && json.error.code) || 'HTTP_ERROR';
          var message = translateError(code);
          if (cbs.onError) {
            cbs.onError(code, message, '');
          }
          cleanup();
        }).catch(function () {
          if (cbs.onError) {
            cbs.onError('HTTP_ERROR', translateError('HTTP_ERROR'), '');
          }
          cleanup();
        });
      }

      // TextDecoder with { stream: true } buffers incomplete multi-byte
      // sequences (e.g. Lithuanian š = 0xC5 0xA1 split across chunks)
      // and combines them with the next chunk. Without this flag,
      // split bytes produce U+FFFD replacement characters.
      var decoder = new TextDecoder('utf-8', { stream: true });
      var reader = response.body.getReader();
      parseSSEStream(reader, decoder, cbs, cleanup);
    }).catch(function (err) {
      if (aborted || (err && err.name === 'AbortError')) {
        cleanup();
        return;
      }
      // Network failure (offline, DNS, CORS)
      if (cbs.onError) {
        var msg = (window.I18n && window.I18n.error_network) || 'Tinklo klaida';
        cbs.onError('NETWORK_ERROR', msg, '');
      }
      cleanup();
    });

    return {
      abort: function () {
        if (aborted) return;
        aborted = true;
        controller.abort();
        cleanup();
      }
    };
  }

  /**
   * Opens a GET SSE connection to /debrief for post-task lesson streaming.
   *
   * Uses the same manual fetch + ReadableStream approach as streamRespond
   * (not EventSource) because the backend requires Authorization header
   * and EventSource can't send custom headers.
   *
   * sessionId: active session ID
   * callbacks: { onToken(text), onDone(fullText, data), onRedact(fallbackText),
   *              onError(code, message, partialText) }
   *
   * Returns { abort } — call abort() to cancel the stream and unlock the UI.
   */
  function streamDebrief(sessionId, callbacks) {
    var cbs = callbacks || {};
    var aborted = false;
    var cleanedUp = false;
    var controller = new AbortController();

    App.updateState({ locked: true });

    var state = App.getState();
    var token = (state.session && state.session.auth_token) || '';

    var headers = {
      'Accept': 'text/event-stream'
    };
    if (token) {
      headers['Authorization'] = 'Bearer ' + token;
    }

    function cleanup() {
      if (cleanedUp) return;
      cleanedUp = true;
      App.updateState({ locked: false });
    }

    fetch(PATHS.debrief(sessionId), {
      method: 'GET',
      headers: headers,
      signal: controller.signal
    }).then(function (response) {
      if (!response.ok) {
        return response.json().then(function (json) {
          var code = (json.error && json.error.code) || 'HTTP_ERROR';
          var message = translateError(code);
          if (cbs.onError) {
            cbs.onError(code, message, '');
          }
          cleanup();
        }).catch(function () {
          if (cbs.onError) {
            cbs.onError('HTTP_ERROR', translateError('HTTP_ERROR'), '');
          }
          cleanup();
        });
      }

      var decoder = new TextDecoder('utf-8', { stream: true });
      var reader = response.body.getReader();
      parseSSEStream(reader, decoder, cbs, cleanup);
    }).catch(function (err) {
      if (aborted || (err && err.name === 'AbortError')) {
        cleanup();
        return;
      }
      if (cbs.onError) {
        var msg = (window.I18n && window.I18n.error_network) || 'Tinklo klaida';
        cbs.onError('NETWORK_ERROR', msg, '');
      }
      cleanup();
    });

    return {
      abort: function () {
        if (aborted) return;
        aborted = true;
        controller.abort();
        cleanup();
      }
    };
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

    // SSE streaming
    streamRespond: streamRespond,
    streamDebrief: streamDebrief,

    // GDPR
    deleteProfile: deleteProfile,
    exportProfile: exportProfile,

    // Radar
    getRadar: getRadar,

    // Utility
    getReport: getReport,
    assetUrl: assetUrl,
    reportClientError: reportClientError,

    // Exposed for SSE functions and future use
    PATHS: PATHS
  };

})();
