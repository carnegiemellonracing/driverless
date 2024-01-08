## Conventions

Conventions for the controls subteam code.

### Naming

Use python naming:

  - ``lower_snake_case`` for values (including functions/methods, 150 moment)
    - Reason: ``lowerCamelCase`` is just ugly
  - ``PascalCase`` for types
    - Reason: prevents declaration issues like ``car car;``
  - Namespace-level declarations should have namespae corresponding to their directory
    - e.g. a declaration in ``src/mppi`` should belong to namespace ``controls::mppi``


### Code Style

  - Follow the cpp core guidelines, except for anything that uses the GSL. Or is otherwise clearly dumb.
  - Misc. notes:
    - use uniform initialization (`{...}` constructor syntax) wherever possible. 
      - Reason: far more
        readable than `(...)` in many cases (looks like a function call/declaration)
    - `m_` prefix should be used for all private fields
      - Reason: disambiguates fields from parameters before it becomes an issue. If you wait until 
        you notice ambiguity, you're already too late!
    - By defualt, prefer immutable data (i.e. `const` lvalues). This also necessitates marking
      methonds `const` wherever possible.
      - Reason: _especially_ since we're multithreading, this makes code easier to reason about. Also
        makes it easier for the compiler to reason about.