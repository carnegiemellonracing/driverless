Pyemojify
=========

|Latest Version| |Build Status| |Python Versions|

Substitutes emoji aliases to emoji raw characters. Simple but sweet
:smile:

Installation
------------

::

    $ pip install pyemojify

Usage
-----

CLI
~~~

Use ``pyemojify -t text``, for example:

::

    $ pyemojify -t "Life is short :smile: , use :sparkles: Python :sparkles:"
    Life is short 😄 , use ✨ Python ✨.

Pyemojify also support pipeline, for example:

::

    $ echo "Life is short :smile: , use :sparkles: Python :sparkles:" | pyemojify
    Life is short 😄 , use ✨ Python ✨.

This one is very useful for git commit messages, use the following one
and you'll see you emoji friends again!

::

    $ git log --oneline --color | pyemojify | less

API
~~~

::

    >>> from pyemojify import emojify
    >>> text = emojify("Life is short :smile: , use :sparkles: Python :sparkles:")
    >>> print(text)
    Life is short 😄 , use ✨ Python ✨.

Credits
-------

It's a python port of the original
`emojify <https://github.com/mrowa44/emojify>`__, all the glories should
belong to `mrowa44 <https://github.com/mrowa44>`__.

License
-------

MIT

.. |Latest Version| image:: http://img.shields.io/pypi/v/pyemojify.svg
   :target: https://pypi.python.org/pypi/pyemojify
.. |Build Status| image:: https://travis-ci.org/lord63/pyemojify.svg
   :target: https://travis-ci.org/lord63/pyemojify
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/pyemojify.svg
   :target: https://pypi.python.org/pypi/pyemojify


