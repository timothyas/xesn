How to Contribute
#################

If you're interested in contributing to xesn, thank you! Here are some pointers
for doing so.

1. Environment Setup
--------------------

First you will want to fork the main repository, and clone that fork onto the
machine where you'll do the development work.
Ultimately, we want any contribution in the form of a pull request that is on
this code fork *on a separate branch from the main branch*.
If this is unfamiliar terminology, check out
`this git tutorial
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_
describing how to make a pull request from a fork, and also
`this page about branches
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches>`_.

After forking the repository, create and activate a development environment as follows::

    cd xesn
    conda env create -f docs/environment.yaml
    conda activate xesn
    pip install -e . --no-deps

and you can test that everything went smoothly by running the unittest suite::

    pytest xesn/test/*.py

2. Develop Contributions
------------------------

Add those awesome contributions to your development branch.
If you are adding a feature to the code base, then make sure to periodically run the test 
suite as shown above::

    cd xesn
    conda activate xesn
    pytest xesn/test/*.py

Additionally, we'll want the new developments to have tests and
`docstrings <https://peps.python.org/pep-0257/>`_
of their own, so
please consider writing tests and documentation during development.

If you are adding to the documentation, then you'll want to first verify that
the documentation builds locally in the environment you created::

    cd xesn/docs
    conda activate xesn
    make html

After that, you can open the generated html files to view in your web browser::

    open build/html/index.html

Rinse and repeat as you add your documentation :)

Don't hesitate to
`create an issue <https://github.com/timothyas/xesn/issues/new>`_
describing the feature or documentation you're interested in adding, and any areas you might like
some help.
We'd be happy to discuss it and help where we can.

3. Submit a Pull Request
------------------------

We recommend doing this on our repository's
`PR webpage 
<https://github.com/timothyas/xesn/pulls>`_
as outlined `on this page
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_.
