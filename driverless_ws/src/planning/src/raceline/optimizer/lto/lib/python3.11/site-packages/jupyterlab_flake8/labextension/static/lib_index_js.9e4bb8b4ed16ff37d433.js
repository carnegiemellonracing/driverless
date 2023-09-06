"use strict";
(self["webpackChunkjupyterlab_flake8"] = self["webpackChunkjupyterlab_flake8"] || []).push([["lib_index_js"],{

/***/ "./lib/index.js":
/*!**********************!*\
  !*** ./lib/index.js ***!
  \**********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/apputils */ "webpack/sharing/consume/default/@jupyterlab/apputils");
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/mainmenu */ "webpack/sharing/consume/default/@jupyterlab/mainmenu");
/* harmony import */ var _jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @jupyterlab/notebook */ "webpack/sharing/consume/default/@jupyterlab/notebook");
/* harmony import */ var _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _jupyterlab_fileeditor__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @jupyterlab/fileeditor */ "webpack/sharing/consume/default/@jupyterlab/fileeditor");
/* harmony import */ var _jupyterlab_fileeditor__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_fileeditor__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _jupyterlab_statedb__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @jupyterlab/statedb */ "webpack/sharing/consume/default/@jupyterlab/statedb");
/* harmony import */ var _jupyterlab_statedb__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_statedb__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var _jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! @jupyterlab/settingregistry */ "webpack/sharing/consume/default/@jupyterlab/settingregistry");
/* harmony import */ var _jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_5___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_5__);
/* harmony import */ var _jupyterlab_terminal__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! @jupyterlab/terminal */ "webpack/sharing/consume/default/@jupyterlab/terminal");
/* harmony import */ var _jupyterlab_terminal__WEBPACK_IMPORTED_MODULE_6___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_terminal__WEBPACK_IMPORTED_MODULE_6__);
/* harmony import */ var _style_index_css__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../style/index.css */ "./style/index.css");







// CSS

// extension id
const id = `jupyterlab-flake8`;
class Preferences {
    constructor() {
        this.toggled = true; // turn on/off linter
        this.logging = false; // turn on/off logging
        this.highlight_color = "var(--jp-warn-color3)"; // color of highlights
        this.gutter_color = "var(--jp-error-color0)"; // color of gutter icons
        this.term_timeout = 5000; // seconds before the temrinal times out if it has not received a message
        this.conda_env = "base"; // conda environment
        this.terminal_name = "flake8term"; // persistent terminal to share between session
        this.configuration_file = ""; // global flake8 configuration file
    }
}
/**
 * Linter
 */
class Linter {
    constructor(app, notebookTracker, editorTracker, palette, mainMenu, state, settingRegistry) {
        this.prefsKey = `${id}:preferences`;
        this.settingsKey = `${id}:plugin`;
        // Default Options
        this.prefs = new Preferences();
        // flags
        this.loaded = false; // flag if flake8 is available
        this.linting = false; // flag if the linter is processing
        this.gutter_id = 'CodeMirror-lintgutter'; // gutter element id
        // cache
        this.marks = []; // text marker objects currently active
        this.bookmarks = []; // text marker objects in editor // --- Temporary fix since gutter doesn't work in editor
        this.docs = []; // text marker objects currently active
        this.text = ''; // current nb text
        this.os = ''; // operating system
        this.app = app;
        this.mainMenu = mainMenu;
        this.notebookTracker = notebookTracker;
        this.editorTracker = editorTracker;
        this.palette = palette;
        this.state = state;
        this.settingRegistry = settingRegistry;
        // load settings from the registry
        Promise.all([
            this.settingRegistry.load(this.settingsKey),
            app.restored,
        ]).then(([settings]) => {
            this.update_settings(settings, true);
            // callback to update settings on changes
            settings.changed.connect((settings) => {
                this.update_settings(settings);
            });
            // on first load, if linter enabled, start it up
            if (this.prefs.toggled) {
                this.load_linter();
            }
        });
        // activate function when cell changes
        this.notebookTracker.currentChanged.connect(this.onActiveNotebookChanged, this);
        // activate when editor changes
        this.editorTracker.currentChanged.connect(this.onActiveEditorChanged, this);
        // add menu item
        this.add_commands();
    }
    /**
     * Update settings callback
     * @param {ISettingRegistry.ISettings} settings
     */
    update_settings(settings, first_load = false) {
        let old = JSON.parse(JSON.stringify(this.prefs)); // copy old prefs
        // set settings to prefs object
        Object.keys(settings.composite).forEach((key) => {
            this.prefs[key] = settings.composite[key];
        });
        this.log(`loaded settings ${JSON.stringify(this.prefs)}`);
        // toggle linter
        if (!first_load && old.toggled !== this.prefs.toggled) {
            this.toggle_linter();
        }
    }
    /**
     * Load terminal session and flake8
     */
    async load_linter() {
        // Bail if there are no terminals available.
        if (!this.app.serviceManager.terminals.isAvailable()) {
            this.log('Disabling jupyterlab-flake8 plugin because it cant access terminal');
            this.loaded = false;
            this.prefs.toggled = false;
            return;
        }
        // try to connect to previous terminal, if not start a new one
        // TODO: still can't set the name of a terminal, so for now saving the "new"
        // terminal name in the settings (#16)
        let session;
        try {
            session = await this.app.serviceManager.terminals
                .connectTo({ model: { name: this.prefs.terminal_name } });
        }
        catch (e) {
            this.log(`starting new terminal session`);
            session = await this.app.serviceManager.terminals.startNew();
        }
        ;
        // save terminal name
        this.setPreference('terminal_name', session.name);
        // start a new terminal session
        this.log(`set terminal_name to ${session.name}`);
        this.term = new _jupyterlab_terminal__WEBPACK_IMPORTED_MODULE_6__.Terminal(session);
        // flush on load
        function _flush_on_load(sender, msg) {
            return;
        }
        // this gets rid of any messages that might get sent on load
        // may fix #28 or #31
        this.term.session.messageReceived.connect(_flush_on_load, this);
        // get OS
        const _this = this;
        function _get_OS(sender, msg) {
            if (msg.content) {
                let message = msg.content[0];
                // throw away non-strings
                if (typeof message !== 'string') {
                    return;
                }
                if (message.indexOf('command not found') > -1) {
                    _this.log(`python command failed on this machine`);
                    _this.term.session.messageReceived.disconnect(_get_OS, _this);
                    _this.finish_load();
                }
                // set OS
                if (message.indexOf('posix') > -1) {
                    _this.os = 'posix';
                }
                else if (message.indexOf('nt(') === -1 &&
                    message.indexOf('int') === -1 &&
                    message.indexOf('nt') > -1) {
                    _this.os = 'nt';
                }
                else {
                    return;
                }
                _this.log(`os: ${_this.os}`);
                // disconnect the os listener and connect empty listener
                _this.term.session.messageReceived.disconnect(_get_OS, _this);
                // setup stage
                _this.setup_terminal();
            }
        }
        // wait a moment for terminal to load and then ask for OS
        setTimeout(() => {
            // disconnect flush
            this.term.session.messageReceived.disconnect(_flush_on_load, this);
            // ask for the OS
            this.term.session.messageReceived.connect(_get_OS, this);
            this.term.session.send({
                type: 'stdin',
                content: [`python -c "import os; print(os.name)"\r`],
            });
        }, 1500);
    }
    setup_terminal() {
        if (this.os === 'posix') {
            this.term.session.send({ type: 'stdin', content: [`HISTFILE= ;\r`] });
        }
        // custom conda-env
        if (this.prefs.conda_env !== 'base') {
            this.set_env();
        }
        else {
            this.finish_load();
        }
    }
    // activate specific conda environment
    set_env() {
        this.log(`conda env: ${this.prefs.conda_env}`);
        if (this.os === 'posix') {
            this.term.session.send({
                type: 'stdin',
                content: [`conda activate ${this.prefs.conda_env}\r`],
            });
        }
        else if (this.os !== 'posix') {
            this.term.session.send({
                type: 'stdin',
                content: [`source activate ${this.prefs.conda_env}\r`],
            });
        }
        this.finish_load();
    }
    finish_load() {
        try {
            // wait a moment for terminal to get initial commands out of its system
            setTimeout(() => {
                this.loaded = true;
                this.activate_flake8();
            }, 500);
        }
        catch (e) {
            this.loaded = false;
            this.prefs.toggled = false;
            this.term.dispose();
        }
    }
    /**
     * Activate flake8 terminal reader
     */
    activate_flake8() {
        // listen for stdout in onLintMessage
        this.term.session.messageReceived.connect(this.onLintMessage, this);
    }
    /**
     * Dispose of the terminal used to lint
     */
    dispose_linter() {
        this.log(`disposing flake8 and terminal`);
        this.lint_cleanup();
        this.clear_marks();
        if (this.term) {
            this.term.session.messageReceived.disconnect(this.onLintMessage, this);
            this.term.dispose();
        }
    }
    /**
     * load linter when notebook changes
     */
    onActiveNotebookChanged() {
        // return if file is being closed
        if (!this.notebookTracker.currentWidget) {
            return;
        }
        // select the notebook
        this.notebook = this.notebookTracker.currentWidget.content;
        this.checkNotebookGutters();
        // run on cell changing
        this.notebookTracker.activeCellChanged.disconnect(this.onActiveCellChanged, this);
        this.notebookTracker.activeCellChanged.connect(this.onActiveCellChanged, this);
        // run on stateChanged
        this.notebook.model.stateChanged.disconnect(this.onActiveCellChanged, this);
        this.notebook.model.stateChanged.connect(this.onActiveCellChanged, this);
    }
    /**
     * Run linter when active cell changes
     */
    onActiveCellChanged() {
        this.checkNotebookGutters();
        if (this.loaded && this.prefs.toggled) {
            if (!this.linting) {
                this.lint_notebook();
            }
            else {
                this.log('flake8 is already running onActiveCellChanged');
            }
        }
    }
    /**
     * load linter when active editor loads
     */
    onActiveEditorChanged() {
        // return if file is being closed
        if (!this.editorTracker.currentWidget) {
            return;
        }
        // select the editor
        this.editor = this.editorTracker.currentWidget.content;
        this.checkEditorGutters();
        // run on stateChanged
        this.editor.model.stateChanged.disconnect(this.onActiveEditorChanges, this);
        this.editor.model.stateChanged.connect(this.onActiveEditorChanges, this);
    }
    /**
     * Run linter on active editor changes
     */
    onActiveEditorChanges() {
        this.checkEditorGutters();
        if (this.loaded && this.prefs.toggled) {
            if (!this.linting) {
                this.lint_editor();
            }
            else {
                this.log('flake8 is already running onEditorChanged');
            }
        }
    }
    checkNotebookGutters() {
        this.notebook.widgets.forEach((widget) => {
            const editor = widget.inputArea.editor;
            const lineNumbers = editor._config.lineNumbers;
            const codeFolding = editor._config.codeFolding;
            const gutters = [
                lineNumbers && 'CodeMirror-linenumbers',
                codeFolding && 'CodeMirror-foldgutter',
                this.gutter_id,
            ].filter((d) => d);
            editor.editor.setOption('gutters', gutters);
        });
    }
    checkEditorGutters() {
        // let editor = this.editorTracker.currentWidget.content;
        // let editorWidget = this.editorTracker.currentWidget;
        const editor = this.editor.editor;
        const lineNumbers = editor._config.lineNumbers;
        const codeFolding = editor._config.codeFolding;
        const gutters = [
            lineNumbers && 'CodeMirror-linenumbers',
            codeFolding && 'CodeMirror-foldgutter',
            this.gutter_id,
        ].filter((d) => d);
        editor.setOption('gutters', gutters);
    }
    /**
     * Generate lint command
     *
     * @param  {string} contents - contents of the notebook ready to be linted
     * @return {string} [description]
     */
    lint_cmd(contents) {
        // escaped characters common to powershell and unix
        let escaped = contents.replace(/[`\\]/g, '\\$&');
        // escaped characters speciic to shell
        if (this.os === 'nt') {
            escaped = contents.replace(/["]/g, '`$&'); // powershell
        }
        else {
            escaped = contents.replace(/["]/g, '\\$&'); // unix
        }
        escaped = escaped.replace('\r', ''); // replace carriage returns
        // ignore magics by commenting
        escaped = escaped
            .split('\n')
            // handle ipy magics %% and %
            .map((line) => {
            if (line.startsWith('%%')) {
                return `# ${line}`;
            }
            else {
                return line;
            }
        })
            .map((line) => {
            if (line.startsWith('%')) {
                return `# ${line}`;
            }
            else {
                return line;
            }
        })
            .join(this.newline());
        // remove final \n (#20)
        if (escaped.endsWith(this.newline())) {
            if (this.os === 'nt') {
                escaped = escaped.slice(0, -2); // powershell
            }
            else {
                escaped = escaped.slice(0, -1); // unix
            }
        }
        let config_option = '';
        if (this.prefs.configuration_file !== null &&
            this.prefs.configuration_file !== '') {
            config_option = `--config="${this.prefs.configuration_file}"`;
        }
        if (this.os === 'nt') {
            // powershell
            return `echo "${escaped}" | flake8 ${config_option} --exit-zero - ; if($?) {echo "@jupyterlab-flake8 finished linting"} ; if (-not $?) {echo "@jupyterlab-flake8 finished linting failed"} `;
        }
        else {
            // unix
            return `(echo "${escaped}" | flake8 ${config_option} --exit-zero - && echo "@jupyterlab-flake8 finished linting" ) || (echo "@jupyterlab-flake8 finished linting failed")`;
        }
    }
    /**
     * Determine new line character based on platform
     */
    newline() {
        // powershell by default on windows
        if (this.os === 'nt') {
            return '`n';
            // otherwise unix
        }
        else {
            return '\n';
        }
    }
    /**
     * Determine if text is input
     * @param {string} text [description]
     */
    text_exists(text) {
        return text;
        // return text && text !== '\n' && text !== '\n\n';
    }
    /**
     * Clear all current marks from code mirror
     */
    clear_marks() {
        // clear marks
        this.marks.forEach((mark) => {
            mark.clear();
        });
        this.marks = [];
        // --- Temporary fix since gutter doesn't work in editor
        // clear error messages in editor
        this.clear_error_messages();
        // clear gutter
        this.docs.forEach((doc) => {
            doc.cm.clearGutter(this.gutter_id);
        });
        this.docs = [];
    }
    /**
     * Lint the CodeMirror Editor
     */
    lint_editor() {
        this.linting = true; // no way to turn this off yet
        this.process_mark = this.mark_editor;
        // catch if file is not a .py file
        if (this.editor.context.path.indexOf('.py') > -1 ||
            this.editor.model._defaultLang === 'python') {
            this.log('getting editor text from python file');
        }
        else {
            this.log(`not python default lang`);
            this.lint_cleanup();
            return;
        }
        let pytext = this.editor.model.value.text;
        this.lint(pytext);
    }
    /**
     * mark the editor pane
     * @param {number} line    [description]
     * @param {number} ch      [description]
     * @param {string} message [description]
     */
    mark_editor(line, ch) {
        this.log(`marking editor`);
        line = line - 1; // 0 index
        ch = ch - 1; // not sure
        // get lines
        let from = { line: line, ch: ch };
        let to = { line: line, ch: ch + 1 };
        // get code mirror editor
        let doc = this.editor.editorWidget.editor.doc;
        return [doc, from, to, 'editor'];
    }
    /**
     * Run flake8 linting on notebook cells
     */
    lint_notebook() {
        this.linting = true; // no way to turn this off yet
        this.process_mark = this.mark_notebook;
        // load notebook
        this.cells = this.notebook.widgets;
        this.log('getting notebook text');
        // return text from each cell if its a code cell
        this.cell_text = this.cells.map((cell, cell_idx, cell_arr) => {
            if (cell.model.type === 'code' &&
                this.text_exists(cell.model.value.text)) {
                // append \n\n if its not the last cell
                if (cell_idx !== cell_arr.length - 1) {
                    return `${cell.model.value.text}\n\n`;
                }
                else {
                    return cell.model.value.text;
                }
            }
            else {
                return '';
            }
        });
        // create dictionary of lines
        this.lookup = {};
        let line = 1;
        this.cell_text.map((cell, cell_idx, cell_arr) => {
            // if there is text in the cell,
            if (this.text_exists(cell)) {
                let lines = cell.split('\n');
                for (let idx = 0; idx < lines.length - 1; idx++) {
                    this.lookup[line] = {
                        cell: cell_idx,
                        line: idx,
                    };
                    line += 1;
                }
            }
            // if its the last cell in the notebook and its empty
            else if (cell_idx === cell_arr.length - 1) {
                this.lookup[line] = {
                    cell: cell_idx,
                    line: 0,
                };
            }
        });
        // ignore other languages (#32)
        // this seems to be all %%magic commands except %%capture
        this.cell_text = this.cell_text.map((cell, cell_idx, cell_arr) => {
            let firstline = cell.split('\n')[0];
            if (firstline && firstline.startsWith("%%") && !(firstline.indexOf("%%capture") > -1)) {
                return cell.split('\n').map((t) => t != "" ? `# ${t}` : "").join('\n');
            }
            else {
                return cell;
            }
        });
        // join cells with text with two new lines
        let pytext = this.cell_text.join('');
        // run linter
        this.lint(pytext);
    }
    /**
     * mark the line of the cell
     * @param {number} line    the line # returned by flake8
     * @param {number} ch      the character # returned by flake 8
     */
    mark_notebook(line, ch) {
        let loc = this.lookup[line];
        ch = ch - 1; // make character 0 indexed
        if (!loc) {
            return;
        }
        let from = { line: loc.line, ch: ch };
        let to = { line: loc.line, ch: ch + 1 };
        // get cell instance
        let cell = this.notebook.widgets[loc.cell];
        // get cell's code mirror editor
        let editor = cell.inputArea.editorWidget
            .editor;
        let doc = editor.doc;
        return [doc, from, to, 'notebook'];
    }
    /**
     * Lint a python text message and callback marking function with line and character
     * @param {string}   pytext        [description]
     */
    lint(pytext) {
        // cache pytext on text
        if (pytext !== this.text) {
            this.text = pytext;
        }
        else {
            // text has not changed
            this.log('text unchanged');
            this.lint_cleanup();
            return;
        }
        // TODO: handle if text is empty (any combination of '' and \n)
        if (!this.text_exists(this.text)) {
            this.log('text empty');
            this.lint_cleanup();
            return;
        }
        // clean current marks
        this.clear_marks();
        // get lint command to run in terminal and send to terminal
        this.log('preparing lint command');
        let lint_cmd = this.lint_cmd(pytext);
        this.log('sending lint command');
        this.term.session.send({ type: 'stdin', content: [`${lint_cmd}\r`] });
        this.termTimeoutHandle = setTimeout(() => {
            if ((this.linting = true)) {
                this.log('lint command timed out');
                alert('jupyterlab-flake8 ran into an issue connecting with the terminal. Please try reloading the browser or re-installing the jupyterlab-flake8 extension.');
                this.lint_cleanup();
                this.dispose_linter();
                this.prefs.toggled = false;
            }
        }, this.prefs.term_timeout);
    }
    /**
     * Handle terminal message during linting
     * TODO: import ISession and IMessage types for sender and msg
     * @param {any} sender [description]
     * @param {any} msg    [description]
     */
    onLintMessage(sender, msg) {
        clearTimeout(this.termTimeoutHandle);
        if (msg.content) {
            let message = msg.content[0];
            // catch non-strings
            if (typeof message !== 'string') {
                return;
            }
            // log message
            this.log(`terminal message: ${message}`);
            // if message a is a reflection of the command, return
            if (message.indexOf('Traceback') > -1) {
                alert(`Flake8 encountered a python error. Make sure flake8 is installed and on the system path. \n\nTraceback: ${message}`);
                this.lint_cleanup();
                return;
            }
            // if message a is a reflection of the command, return
            if (message.indexOf('command not found') > -1) {
                alert(`Flake8 was not found in this python environment. \n\nIf you are using a conda environment, set the 'conda_env' setting in the Advanced Settings menu and reload the Jupyter Lab window.\n\nIf you are not using a conda environment, Install Flake8 with 'pip install flake8' or 'conda install flake8' and reload the Jupyter Lab window`);
                this.lint_cleanup();
                return;
            }
            message.split(/(?:\n|\[)/).forEach((m) => {
                if (m.includes('stdin:')) {
                    let idxs = m.split(':');
                    let line = parseInt(idxs[1]);
                    let ch = parseInt(idxs[2]);
                    this.log(idxs[3]);
                    this.get_mark(line, ch, idxs[3].slice(0, -1));
                }
            });
            if (message.indexOf('jupyterlab-flake8 finished linting') > -1) {
                this.lint_cleanup();
            }
        }
    }
    /**
     * Mark a line in notebook or editor
     * @param {number} line    [description]
     * @param {number} ch      [description]
     * @param {string} message [description]
     */
    get_mark(line, ch, message) {
        let doc, from, to, context;
        try {
            if (this.process_mark && typeof this.process_mark === 'function') {
                [doc, from, to, context] = this.process_mark(line, ch);
            }
        }
        catch (e) {
            this.log(`failed to run process_mark`);
            return;
        }
        if (!doc || !from || !to) {
            this.log(`mark location not fully defined`);
            return;
        }
        this.mark_line(doc, from, to, message, context);
    }
    /**
     * Mark line in document
     * @param {any}    doc     [description]
     * @param {any}    from    [description]
     * @param {any}    to      [description]
     * @param {string} message [description]
     */
    mark_line(doc, from, to, message, context) {
        let gutter_color = this.prefs.gutter_color;
        // gutter marker - this doesn't work in the editor
        function makeMarker() {
            let marker = document.createElement('div');
            marker.innerHTML = `<div class='jupyterlab-flake8-lint-gutter-container' style='color: ${gutter_color}''>
        <div>◉</div><div class='jupyterlab-flake8-lint-gutter-message'>${message}</div>
      </div>`;
            return marker;
        }
        // store gutter marks for later
        doc.cm.setGutterMarker(from.line, this.gutter_id, makeMarker());
        this.docs.push(doc);
        // --- Temporary fix since gutters don't show up in editor
        // show error message in editor
        if (context === 'editor') {
            let lint_alert = document.createElement('span');
            let lint_message = document.createTextNode(`------ ${message}`);
            lint_alert.appendChild(lint_message);
            lint_alert.className = 'jupyterlab-flake8-lint-message-inline';
            // add error alert node to the 'to' location
            this.bookmarks.push(doc.addLineWidget(from.line, lint_alert));
        }
        // mark the text position with highlight
        this.marks.push(doc.markText(from, to, {
            // replacedWith: selected_char_node,
            className: 'jupyterlab-flake8-lint-message',
            css: `
          background-color: ${this.prefs.highlight_color}
        `,
        }));
    }
    /**
     * // --- Temporary fix since gutters don't show up in editor
     * Clear all error messages
     */
    clear_error_messages() {
        this.bookmarks.forEach((bookmark) => {
            bookmark.clear();
        });
    }
    /**
     * Tear down lint fixtures
     */
    lint_cleanup() {
        this.linting = false;
        // this.process_mark = undefined;
    }
    /**
     * Show browser logs
     * @param {any} msg [description]
     */
    log(msg) {
        // return if prefs.logging is not enabled
        if (!this.prefs.logging) {
            return;
        }
        // convert object messages to strings
        if (typeof msg === 'object') {
            msg = JSON.stringify(msg);
        }
        // prepend name
        let output = `jupyterlab-flake8: ${msg}`;
        console.log(output);
    }
    /**
     * Create menu / command items
     */
    add_commands() {
        let category = 'Flake8';
        // define all commands
        let commands = {
            'flake8:toggle': {
                label: 'Enable Flake8',
                isEnabled: () => {
                    return this.loaded;
                },
                isToggled: () => {
                    return this.prefs.toggled;
                },
                execute: async () => {
                    this.setPreference('toggled', !this.prefs.toggled);
                },
            },
            'flake8:show_browser_logs': {
                label: 'Output Flake8 Browser Console Logs',
                isEnabled: () => {
                    return this.loaded;
                },
                isToggled: () => {
                    return this.prefs.logging;
                },
                execute: () => {
                    this.setPreference('logging', !this.prefs.logging);
                },
            },
        };
        // add commands to menus and palette
        for (let key in commands) {
            this.app.commands.addCommand(key, commands[key]);
            this.palette.addItem({ command: key, category: category });
        }
        // add to view Menu
        this.mainMenu.viewMenu.addGroup(Object.keys(commands).map((key) => {
            return { command: key };
        }), 30);
    }
    /**
     * Turn linting on/off
     */
    toggle_linter() {
        if (this.prefs.toggled) {
            this.load_linter();
        }
        else {
            this.dispose_linter();
        }
    }
    /**
     * Save state preferences
     */
    async setPreference(key, val) {
        await Promise.all([
            this.settingRegistry.load(this.settingsKey),
            this.app.restored,
        ]).then(([settings]) => {
            settings.set(key, val); // will automatically call update
        });
    }
}
/**
 * Activate extension
 */
function activate(app, notebookTracker, editorTracker, palette, mainMenu, state, settingRegistry) {
    new Linter(app, notebookTracker, editorTracker, palette, mainMenu, state, settingRegistry);
}
// activate: (app: JupyterFrontEnd, settingRegistry: ISettingRegistry | null) => {
//   console.log('JupyterLab extension jupyterlab-flake8 is activated!');
//   if (settingRegistry) {
//     settingRegistry
//       .load(plugin.id)
//       .then(settings => {
//         console.log('jupyterlab-flake8 settings loaded:', settings.composite);
//       })
//       .catch(reason => {
//         console.error('Failed to load settings for jupyterlab-flake8.', reason);
//       });
//   }
// }
/**
 * Initialization data for the jupyterlab-flake8 extension.
 */
const plugin = {
    id: 'jupyterlab-flake8',
    autoStart: true,
    activate: activate,
    requires: [
        _jupyterlab_notebook__WEBPACK_IMPORTED_MODULE_2__.INotebookTracker,
        _jupyterlab_fileeditor__WEBPACK_IMPORTED_MODULE_3__.IEditorTracker,
        _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.ICommandPalette,
        _jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_1__.IMainMenu,
        _jupyterlab_statedb__WEBPACK_IMPORTED_MODULE_4__.IStateDB,
        _jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_5__.ISettingRegistry
    ],
};
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (plugin);


/***/ }),

/***/ "./node_modules/css-loader/dist/cjs.js!./style/base.css":
/*!**************************************************************!*\
  !*** ./node_modules/css-loader/dist/cjs.js!./style/base.css ***!
  \**************************************************************/
/***/ ((module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _node_modules_css_loader_dist_runtime_cssWithMappingToString_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../node_modules/css-loader/dist/runtime/cssWithMappingToString.js */ "./node_modules/css-loader/dist/runtime/cssWithMappingToString.js");
/* harmony import */ var _node_modules_css_loader_dist_runtime_cssWithMappingToString_js__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_node_modules_css_loader_dist_runtime_cssWithMappingToString_js__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _node_modules_css_loader_dist_runtime_api_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../node_modules/css-loader/dist/runtime/api.js */ "./node_modules/css-loader/dist/runtime/api.js");
/* harmony import */ var _node_modules_css_loader_dist_runtime_api_js__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_node_modules_css_loader_dist_runtime_api_js__WEBPACK_IMPORTED_MODULE_1__);
// Imports


var ___CSS_LOADER_EXPORT___ = _node_modules_css_loader_dist_runtime_api_js__WEBPACK_IMPORTED_MODULE_1___default()((_node_modules_css_loader_dist_runtime_cssWithMappingToString_js__WEBPACK_IMPORTED_MODULE_0___default()));
// Module
___CSS_LOADER_EXPORT___.push([module.id, ".jupyterlab-flake8-lint-message span {\n  display: none;\n}\n\n.jupyterlab-flake8-lint-message span:hover {\n  display: block;\n}\n\n.CodeMirror-lintgutter {\n  padding: 0 8px;\n}\n\n.jupyterlab-flake8-lint-gutter-container {\n  cursor: pointer;\n  display: flex;\n  justify-content: center;\n  position: relative;\n}\n\n.jupyterlab-flake8-lint-gutter-message {\n  background: var(--jp-layout-color4);\n  border-radius: 4px;\n  color: white;\n  left: 16px;\n  padding: 2px 5px;\n  position: absolute;\n  top: 0;\n  visibility: hidden;\n  white-space: nowrap;\n  z-index: 1;\n}\n\n.jupyterlab-flake8-lint-gutter-container:hover\n  .jupyterlab-flake8-lint-gutter-message {\n  visibility: visible;\n}\n", "",{"version":3,"sources":["webpack://./style/base.css"],"names":[],"mappings":"AAAA;EACE,aAAa;AACf;;AAEA;EACE,cAAc;AAChB;;AAEA;EACE,cAAc;AAChB;;AAEA;EACE,eAAe;EACf,aAAa;EACb,uBAAuB;EACvB,kBAAkB;AACpB;;AAEA;EACE,mCAAmC;EACnC,kBAAkB;EAClB,YAAY;EACZ,UAAU;EACV,gBAAgB;EAChB,kBAAkB;EAClB,MAAM;EACN,kBAAkB;EAClB,mBAAmB;EACnB,UAAU;AACZ;;AAEA;;EAEE,mBAAmB;AACrB","sourcesContent":[".jupyterlab-flake8-lint-message span {\n  display: none;\n}\n\n.jupyterlab-flake8-lint-message span:hover {\n  display: block;\n}\n\n.CodeMirror-lintgutter {\n  padding: 0 8px;\n}\n\n.jupyterlab-flake8-lint-gutter-container {\n  cursor: pointer;\n  display: flex;\n  justify-content: center;\n  position: relative;\n}\n\n.jupyterlab-flake8-lint-gutter-message {\n  background: var(--jp-layout-color4);\n  border-radius: 4px;\n  color: white;\n  left: 16px;\n  padding: 2px 5px;\n  position: absolute;\n  top: 0;\n  visibility: hidden;\n  white-space: nowrap;\n  z-index: 1;\n}\n\n.jupyterlab-flake8-lint-gutter-container:hover\n  .jupyterlab-flake8-lint-gutter-message {\n  visibility: visible;\n}\n"],"sourceRoot":""}]);
// Exports
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (___CSS_LOADER_EXPORT___);


/***/ }),

/***/ "./node_modules/css-loader/dist/cjs.js!./style/index.css":
/*!***************************************************************!*\
  !*** ./node_modules/css-loader/dist/cjs.js!./style/index.css ***!
  \***************************************************************/
/***/ ((module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _node_modules_css_loader_dist_runtime_cssWithMappingToString_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../node_modules/css-loader/dist/runtime/cssWithMappingToString.js */ "./node_modules/css-loader/dist/runtime/cssWithMappingToString.js");
/* harmony import */ var _node_modules_css_loader_dist_runtime_cssWithMappingToString_js__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_node_modules_css_loader_dist_runtime_cssWithMappingToString_js__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _node_modules_css_loader_dist_runtime_api_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../node_modules/css-loader/dist/runtime/api.js */ "./node_modules/css-loader/dist/runtime/api.js");
/* harmony import */ var _node_modules_css_loader_dist_runtime_api_js__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_node_modules_css_loader_dist_runtime_api_js__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _node_modules_css_loader_dist_cjs_js_base_css__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! -!../node_modules/css-loader/dist/cjs.js!./base.css */ "./node_modules/css-loader/dist/cjs.js!./style/base.css");
// Imports



var ___CSS_LOADER_EXPORT___ = _node_modules_css_loader_dist_runtime_api_js__WEBPACK_IMPORTED_MODULE_1___default()((_node_modules_css_loader_dist_runtime_cssWithMappingToString_js__WEBPACK_IMPORTED_MODULE_0___default()));
___CSS_LOADER_EXPORT___.i(_node_modules_css_loader_dist_cjs_js_base_css__WEBPACK_IMPORTED_MODULE_2__["default"]);
// Module
___CSS_LOADER_EXPORT___.push([module.id, "\n", "",{"version":3,"sources":[],"names":[],"mappings":"","sourceRoot":""}]);
// Exports
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (___CSS_LOADER_EXPORT___);


/***/ }),

/***/ "./style/index.css":
/*!*************************!*\
  !*** ./style/index.css ***!
  \*************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _node_modules_style_loader_dist_runtime_injectStylesIntoStyleTag_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! !../node_modules/style-loader/dist/runtime/injectStylesIntoStyleTag.js */ "./node_modules/style-loader/dist/runtime/injectStylesIntoStyleTag.js");
/* harmony import */ var _node_modules_style_loader_dist_runtime_injectStylesIntoStyleTag_js__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_node_modules_style_loader_dist_runtime_injectStylesIntoStyleTag_js__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _node_modules_css_loader_dist_cjs_js_index_css__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! !!../node_modules/css-loader/dist/cjs.js!./index.css */ "./node_modules/css-loader/dist/cjs.js!./style/index.css");

            

var options = {};

options.insert = "head";
options.singleton = false;

var update = _node_modules_style_loader_dist_runtime_injectStylesIntoStyleTag_js__WEBPACK_IMPORTED_MODULE_0___default()(_node_modules_css_loader_dist_cjs_js_index_css__WEBPACK_IMPORTED_MODULE_1__["default"], options);



/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (_node_modules_css_loader_dist_cjs_js_index_css__WEBPACK_IMPORTED_MODULE_1__["default"].locals || {});

/***/ })

}]);
//# sourceMappingURL=lib_index_js.9e4bb8b4ed16ff37d433.js.map