.. _install:

Installation
============

System wide deployment
----------------------

If you have administrator privileges on the machine, you can install `tcvpy`
system-wide, making it available to all users::

    $ sudo pip install https://github.com/wagdav/tcvpy/zipball/master

Installing in $HOME
-------------------

As a non-privileged user you can install the `tcvpy` in directly in your home
directory::

    $ pip install --user https://github.com/wagdav/tcvpy/zipball/master

Or, using `virtualenv`, you can fully isolate your installation::

    $ virtualenv tcvpy
    $ source tcvpy/bin/activate
    (tcvpy)$ pip install https://github.com/wagdav/tcvpy/zipball/master

Checking the installation
-------------------------

If everything goes well, the following command should not give any error::

    $ python -c 'import tcv'

**Note**: if you chose to install the package with `virtualenv`, you should
have the corresponding virtual environment activated.

What's next
-----------

Check out the example in the :py:mod:`tcv` module's documentation.
