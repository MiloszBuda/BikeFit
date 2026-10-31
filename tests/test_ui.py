def test_ui_import():
    from bikefit.ui.app import main, run_app
    assert callable(main)
    assert callable(run_app)
