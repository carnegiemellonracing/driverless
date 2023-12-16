## Conventions

Conventions for the controls subteam code.

### Naming

Use ``boost``/``STL`` naming.

  - ``lower_snake_case`` for everything except mentioned below
  - ``Upper_Snake_Case`` for type parameters

Reasoning: consistent with standard library. incredibly simple.


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