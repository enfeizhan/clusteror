from setuptools_scm.utils import do_ex, do


def test_pkginfo_noscmroot(tmpdir, monkeypatch):
    """if we are indeed a sdist, the root does not apply"""
    monkeypatch.delenv("SETUPTOOLS_SCM_DEBUG")

    # we should get the version from pkg-info if git is broken
    p = tmpdir.ensure('sub/package', dir=1)
    tmpdir.mkdir('.git')
    p.join('setup.py').write(
        'from setuptools import setup;'
        'setup(use_scm_version={"root": ".."})')

    _, stderr, ret = do_ex('python setup.py --version', p)
    assert 'setuptools-scm was unable to detect version for' in stderr
    assert ret == 1

    p.join("PKG-INFO").write('Version: 1.0')
    res = do('python setup.py --version', p)
    assert res == '1.0'

    do('git init', p.dirpath())
    res = do('python setup.py --version', p)
    assert res == '1.0'


def test_use_scm_version_callable(tmpdir, monkeypatch):
    """use of callable as use_scm_version argument"""
    monkeypatch.delenv("SETUPTOOLS_SCM_DEBUG")

    p = tmpdir.ensure('sub/package', dir=1)
    p.join('setup.py').write(
        '''from setuptools import setup
def vcfg():
    from setuptools_scm.version import guess_next_dev_version
    def vs(v):
        return guess_next_dev_version(v)
    return {"version_scheme": vs}
setup(use_scm_version=vcfg)
''')
    p.join("PKG-INFO").write('Version: 1.0')

    res = do('python setup.py --version', p)
    assert res == '1.0'
