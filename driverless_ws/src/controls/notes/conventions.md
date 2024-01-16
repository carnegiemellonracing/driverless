## Conventions

Conventions for the controls subteam code, _in general_. Specific exceptions for `.cu`/`.cuh` can be found in _how_to_cuda.md_.

### Philosophy
* Write simply and readably first, then optimize
  * Reason: often we won't need to optimize, and simpler code is less buggy and more extensible
* Separate CUDA from C++ as much as possible
  * Reason: CUDA is gross. And buggy. And not object-oriented. Good encapsulation will help contain the 
    jankness as much as possible.
* Follow the cpp core guidelines as a starting point, but don't be afraid to diverge if you feel something is
  particularly inconvenient 
* Document _everything_. 
  * And by everything we mean:
    * Any namespace-scope declaration
    * Methods
    * Fields
    * Opaque implementation details
  * And by document we mean:
    * Not picky about format, but make it an established docstring format (e.g. Doxygen) 
      so hovering works in IDEs
    * The **what** concisely up top, the **how** if it is non-obvious or has performance implications, 
      and the **why** if you feel it is necessary
    * Pre/Postconditions that aren't enforced by type-checking
    * Parameters and return values even if it feels redundant so hovering works properly

### Naming

Use python naming:

  - ``lower_snake_case`` for values (including functions/methods, 150 moment)
    - Reason: consistency with STL and CUDA
  - ``PascalCase`` for types
    - Reason: prevents declaration grossness like ``car car;``
  - Namespace-level declarations should have namespace corresponding to their directory
    - e.g. a declaration in ``src/mppi`` should belong to namespace ``controls::mppi``


### Style

  - use uniform initialization (`{...}` constructor syntax) wherever possible. 
    - Reason: far more
      readable than `(...)` in many cases (looks like a function call/declaration)
  - `m_` prefix should be used for all private fields
    - Reason: disambiguates fields from parameters before it becomes an issue. If you wait until 
      you notice ambiguity, you're already too late!
  - By default, prefer immutable data (i.e. `const` lvalues). This also necessitates marking
    methods `const` wherever possible.
    - Reason: _especially_ since we're multithreading, this makes code easier to reason about. Also
      makes it easier for the compiler to reason about.