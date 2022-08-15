# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/john-p/wsic/issues>.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in
  troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and
"help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
"enhancement" and "help wanted" is open to whoever wants to implement
it.

### Write Documentation

wsic could always use more documentation, whether as part of the
official wsic docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/john-p/wsic/issues>.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to
  implement.
- Remember that this is a volunteer-driven project, and that
  contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up
<span class="title-ref">wsic</span> for local development.

1. Fork the <span class="title-ref">wsic</span> repo on GitHub.

1. Clone your fork locally:

   ```shell
   $ git clone git@github.com:your_name_here/wsic.git
   ```

1. Install your local copy into a virtualenv. Assuming you have
   virtualenvwrapper installed, this is how you set up your fork for
   local development:

   ```shell
   $ mkvirtualenv wsic
   $ cd wsic/
   $ python setup.py develop
   ```

1. Create a branch for local development:

   ```shell
   $ git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

1. When you're done making changes, check that your changes pass flake8
   and the tests, including testing other Python versions with tox:

   ```shell
   $ flake8 wsic tests
   $ python setup.py test or pytest
   $ tox
   ```

   To get flake8 and tox, just pip install them into your virtualenv.

1. Commit your changes and push your branch to GitHub:

   ```shell
   $ git add .
   $ git commit -m "Your detailed description of your changes."
   $ git push origin name-of-your-bugfix-or-feature
   ```

1. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
1. If the pull request adds functionality, the docs should be updated.
   Put your new functionality into a function with a docstring, and add
   the feature to the list in README.rst.
1. The pull request should work for cPython 3.8, 3.9, 3.10. Check the
   GitHib Actions workflow on the pull request and make sure that the
   tests pass for all supported Python versions.

## Tips

To run a subset of tests:

```shell
$ pytest tests.test_wsic
```

## Deploying

A reminder for the maintainers on how to deploy. Make sure all your
changes are committed (including an entry in HISTORY.md). Then run:

```shell
$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```

GitHub Actions will then deploy to PyPI if tests pass.
