from argparse import Namespace
import numpy as np
import os
import shutil

from ..em import em_parsing, lightcurve_handling as lch
from ..em.io import load_em_observations
from ..em.model import get_lc_model_from_modelname
from ..core.utils import read_injection_file
from ..core.parsing import nmma_base_parsing
from ..core.conversion import BNSEjectaFitting, KilonovaEjectaFitting
from ..joint import injection_handling, joint_parsing


def lightcurveInjectionTest(model_name):
    """
    compares the creation of a lightcurve injection from command line with light_curve_generation and through calling the relevant function directly
    Parameters:
    -----------
    - model_name: string
    Name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
    """
    print("running lightcurve injection test for ", model_name)
    print(
        "current working directory: ", os.getcwd()
    )  # assumes run in root nmma folder, will need to modify if this is not true
    workingDir = os.path.dirname(__file__)
    dataDir = os.path.join(workingDir, "data")
    test_directory = os.path.join(dataDir, model_name)
    priorDir = os.path.join(workingDir, "../../priors/")
    svdmodels = os.path.join(workingDir, "../../svdmodels/")
    if os.path.isdir(test_directory):
        shutil.rmtree(test_directory, ignore_errors=True)
    os.makedirs(test_directory, exist_ok=True)

    def create_injection_from_command_line(model_name):
        """
        Creates the injection file from command line using nmma_create_injection
        Parameters:
        ------------
        - model_name: string
        Name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory

        Returns:
        ---------
        - injection_name: string
        path to the injection file created by nmma_create_injection
        """

        if model_name == "nugent-hyper":
            prior_path = os.path.join(priorDir, "sncosmo-generic" + ".prior")
        elif model_name == "TrPi2018":
            prior_path = os.path.join(
                dataDir, "TrPi2018_pinned_parameters" + ".prior"
            )  # pinning the parameter svalues in the prior file
        else:
            prior_path = os.path.join(priorDir, model_name + ".prior")
        assert os.path.exists(prior_path), "prior file does not exist"
        injection_name = os.path.join(test_directory, model_name + "_injection.json")

        args = nmma_base_parsing(joint_parsing.injection_parsing, [])
        non_default_args = dict(
            prior_file=prior_path,
            simple_setup=True,
            injection_file=injection_name,
            post_processing=["ejecta"],
            n_injection=1,
            eos_file="example_files/eos/ALF2.dat",
            original_parameters=True,
        )

        for key, value in non_default_args.items():
            setattr(args, key, value)
        injection_handling.generate_injection(args)

        assert os.path.exists(injection_name), "injection file does not exist"
        return injection_name

    def create_lightcurve_from_command_line(model_name, injection_file):
        """
        Creates the lightcurve file from command line using light_curve_generation
        Parameters:
        ------------
        - model_name: string
        Name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
        - injection_file: string
        path to injection file created by create_injection_from_command_line

        Returns:
        ----------
        - command_line_lightcurve_file: string
        path to the lightcurve file created by light_curve_generation
        """
        # prior_path = os.path.join("./priors/", model_name + ".prior")
        output_directory = test_directory
        command_line_lightcurve_label = model_name + "_command_line"

        args = em_parsing.parsing_and_logging(em_parsing.lightcurve_parser, [])
        non_default_args = dict(
            injection_file=injection_file,
            label=command_line_lightcurve_label,
            em_model=model_name,
            svd_path=svdmodels,
            em_tmin=0.01,
            em_tmax=20.0,
            em_tstep=0.5,
            filters="sdssu",
            outdir=output_directory,
            interpolation_type="sklearn_gp",
            injection_error_budget=0.0,
            ignore_timeshift=True,
        )
        args.__dict__.update(non_default_args)

        lch.lcs_from_injection_parameters(args)

        command_line_lightcurve_file = os.path.join(
            output_directory, f"{command_line_lightcurve_label}_0_lc.json"
        )
        assert os.path.exists(
            command_line_lightcurve_file
        ), "command line lightcurve file does not exist"

        return load_em_observations(command_line_lightcurve_file)

    def create_lightcurve_from_function(model_name, injection_file):
        """
        create lightcurve using associated LightcurveModel function
        Parameters:
        ------------
        - model_name: string
        name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
        - injection_file: string
        path to injection file
        Returns:
        ----------
        - lightcurve_from_function: dictionary
        dictionary of lightcurve generated via functions
        """
        assert os.path.exists(injection_file), "injection file does not exist"
        injection_dict = read_injection_file(injection_file)
        lightcurve_parameters = {k: v[0] for k, v in injection_dict.items()}
        init_kwargs = dict(
            model=model_name,
            filters=["sdssu"],
            sample_times=np.arange(0.01, 20.0 + 0.5, 0.5),
        )
        if model_name == "Ka2017":
            init_kwargs["interpolation_type"] = "sklearn_gp"
        model_class = get_lc_model_from_modelname(model_name)
        lightcurve_model = model_class(**init_kwargs)
        lc_params = lightcurve_model.parameter_conversion(lightcurve_parameters)
        _, func_lc = lightcurve_model.gen_detector_lc(lc_params)
        # lightcurve_from_function["t"] = obs_times

        return func_lc

    def compare_lightcurves(lightcurve_from_function, lightcurve_from_command_line):
        """
        Compare the values of the lightcurves generated from the function and command line to look for differences

        Parameters:
        ------------
        - lightcurve_from_function: dictionary
        Dictionary of lightcurve generated from function (keys: filters, values: list of magnitudes)
        - lightcurve_from_command_line: dictionary
        Dictionary of lightcurve generated from command line (keys: filters, values: list of magnitudes)

        Returns:
        None
        """
        filters_from_function = lightcurve_from_function.keys()
        filters_from_command_line = lightcurve_from_command_line.keys()

        assert set(filters_from_function) == set(
            filters_from_command_line
        ), "filters from function and command line do not match"
        # goes filter by filter and checks that each array matches
        for filter_name in filters_from_function:
            cli_mags = lightcurve_from_command_line[filter_name]["mag"]
            gen_mags = lightcurve_from_function[filter_name]
            assert all(
                np.isclose(
                    cli_mags[~np.isnan(cli_mags)],
                    gen_mags[~np.isnan(gen_mags)],
                    rtol=1e-3,
                )
            ), f"lightcurve tolerance for {filter_name} exceeded"

    def cleanup_files():
        """
        deletes test files directory
        """
        shutil.rmtree(test_directory, ignore_errors=True)
        assert not os.path.exists(test_directory), "test directory has not been deleted"

    injection_file = create_injection_from_command_line(model_name)
    command_line_lightcurve_dictionary = create_lightcurve_from_command_line(
        model_name, injection_file
    )
    function_lightcurve_dictionary = create_lightcurve_from_function(
        model_name, injection_file
    )

    compare_lightcurves(
        function_lightcurve_dictionary, command_line_lightcurve_dictionary
    )

    # if all of the above works, then we don't need the files anymore
    cleanup_files()


def test_injections():
    for model_name in ["nugent-hyper", "salt2", "Me2017", "Piro2021", "TrPi2018"]:
        lightcurveInjectionTest(model_name)


def test_eos_injection_without_snr_test():
    """FIXME Weizmann: regression test, using the real Bu2019lm.prior +
    example_files/sim_events/bns_O4_injections.dat combination reported in
    the original bug (nmma-create-injection --eos-file ... --gw-injection-file
    ... with no --tests/--post-processing snr), for bugs in
    NMMAInjectionCreator that only trigger when simple_setup=False (i.e.
    setup_test_routines/test_wrap actually run, unlike the other tests in
    this file, which all use simple_setup=True and so never exercised this
    code path):

    1. EoSConverter needs mass_1_source/mass_2_source, but the 'gw' step
       that computes them (bbh_source_frame) used to only get added to
       conv_instructions for an 'snr' test/post-processing. Using
       --eos-file without an snr test raised
       KeyError('mass_1_source') in EoSConverter.system_props_from_eos.
    2. test_wrap() used to call self.priors.evaluate_constraints(test_df)
       with a pandas DataFrame. bilby's evaluate_constraints does
       next(iter(sample.values())) to get a template array; DataFrame.values
       is an ndarray property, so calling it raises TypeError, silently
       caught by evaluate_constraints, which then falls back to
       np.ones_like(sample) over the whole 2D table instead of a 1D
       per-row array, raising
       ValueError: Expected a 1D array, got an array with shape (N, n_cols)
       when assigned back into test_df['tests_passed'].
    3. refill_failed_tests() did retest_df["tests_passed"] = self.test_wrap(
       retest_df), but test_wrap() returns the whole dataframe, not just
       that column, raising ValueError: Columns must be same length as key.
    4. testing_and_postprocessing()'s redraw branch only copied the
       'tests_passed' column back onto the pre-conversion dataframe, and
       refill_failed_tests() only copied 'tests_passed' back for redrawn
       rows: with --original-parameters (which skips the final
       core_conversion pass), mass_1_source/lambda_1/2/radius_1/2 ended up
       silently missing from the output whenever at least one row needed a
       redraw.
    """
    workingDir = os.path.dirname(__file__)
    dataDir = os.path.join(workingDir, "data")
    test_directory = os.path.join(dataDir, "eos_injection_no_snr_test")
    priorDir = os.path.join(workingDir, "../../priors/")
    exampleFilesDir = os.path.join(workingDir, "../../example_files/")
    if os.path.isdir(test_directory):
        shutil.rmtree(test_directory, ignore_errors=True)
    os.makedirs(test_directory, exist_ok=True)

    prior_path = os.path.join(priorDir, "Bu2019lm.prior")
    assert os.path.exists(prior_path), "prior file does not exist"

    # bns_O4_injections.dat has 2000 rows and some masses well above ALF2's
    # ~2.09 Msun TOV mass (unphysical for this EOS, and slow to run through
    # in full for a unit test); keep only a couple of rows with masses
    # safely inside ALF2's range.
    full_dat_path = os.path.join(exampleFilesDir, "sim_events", "bns_O4_injections.dat")
    assert os.path.exists(full_dat_path), "example gw-injection-file does not exist"
    gw_injection_path = os.path.join(test_directory, "bns_O4_injections_subset.dat")
    with open(full_dat_path) as f:
        lines = f.readlines()
    header, rows = lines[0], lines[1:]
    selected = [
        row
        for row in rows
        if all(float(x) < 2.0 for x in row.split("\t")[5:7])  # mass1, mass2
    ][:2]
    assert len(selected) == 2, "could not find 2 rows with safe masses"
    with open(gw_injection_path, "w") as f:
        f.writelines([header] + selected)

    injection_name = os.path.join(test_directory, "eos_injection.json")

    args = nmma_base_parsing(joint_parsing.injection_parsing, [])
    non_default_args = dict(
        prior_file=prior_path,
        gw_injection_file=gw_injection_path,
        injection_file=injection_name,
        eos_file="example_files/eos/ALF2.dat",
        original_parameters=True,
        generation_seed=42,
    )
    for key, value in non_default_args.items():
        setattr(args, key, value)

    # simple_setup defaults to False here: this must go through
    # setup_test_routines/testing_and_postprocessing/test_wrap, not skip them.
    injection_handling.generate_injection(args)

    assert os.path.exists(injection_name), "injection file does not exist"
    injection_dict = read_injection_file(injection_name)
    for key in (
        "mass_1_source",
        "mass_2_source",
        "lambda_1",
        "lambda_2",
        "radius_1",
        "radius_2",
    ):
        assert key in injection_dict, f"{key} missing from generated injection"

    shutil.rmtree(test_directory, ignore_errors=True)


def test_validate_lightcurves():
    print("validate_lightcurve test")

    # initialize args, check a file that is known to have 3 observations in the ztf g filter and 1 in the ztf r filter. All detections occur within 9 days of the original observation.
    args = Namespace(
        data_file="example_files/candidate_data/ZTF20abwysqy.dat",
        filters=["ztfg"],
        min_obs=3,
        cutoff_time=0,
        verbose=True,
    )
    assert lch.validate_lightcurve(
        **vars(args)
    ), "Test for 3 observations in the ztf g filter failed"

    args.filters = ["ztfr"]
    args.min_obs = 1
    assert lch.validate_lightcurve(
        **vars(args)
    ), "Test for 1 observation in the ztf r filter failed"

    args.filters = ["ztfg", "ztfr"]
    assert lch.validate_lightcurve(
        **vars(args)
    ), "Test for  passing multiple filters failed"

    args.filters = None
    args.min_obs = 0
    assert lch.validate_lightcurve(
        **vars(args)
    ), "Test for automatic filter selection failed"

    args.cutoff_time = 1
    args.min_obs = 1
    assert not lch.validate_lightcurve(
        **vars(args)
    ), "Test for setting cutoff time failed"


def test_bns_ejecta_conversion_rejects_non_ns_component():
    """FIXME Weizmann: regression test for a bug in BNSEjectaFitting.bns_ejecta_conversion.

    radius_1/radius_2 are 0 (not a real Schwarzschild radius) for a mass
    outside the EOS's tabulated range, so compactness = mass*geom_msun_km/radius
    is inf for that component. dynamic_mass_fitting_KrFo/log10_disk_mass_fitting
    clip negative fit values via np.maximum(0, .), which silently turned that
    inf into a finite mdyn_fit=0.0 before np.isfinite() downstream had any
    chance to notice, log10(0 + alpha) then came out as an ordinary finite
    number for a system that isn't actually a BNS under this EOS. Verified
    directly against the exact formula: with mass_1_source=2.5 (above ALF2's
    ~2.09 Msun TOV mass) and radius_1=0, this used to return a finite
    log10_mej_dyn instead of -inf.
    """
    fitter = BNSEjectaFitting()
    params = dict(
        mass_1_source=np.array([2.5]),
        mass_2_source=np.array([1.3]),
        radius_1=np.array([0.0]),  # outside ALF2's mass range -> not a NS
        radius_2=np.array([13.1]),
        alpha=np.array([0.04]),
        ratio_zeta=np.array([0.5]),
        TOV_mass=np.array([2.0854]),
        R_16=np.array([12.0]),
    )
    log10_mej_dyn, log10_mej_wind, log10_mej_total, _ = fitter.bns_ejecta_conversion(
        params
    )
    assert not np.isfinite(
        log10_mej_dyn[0]
    ), "log10_mej_dyn should be -inf when mass_1 isn't a real NS under this EOS"
    assert not np.isfinite(
        log10_mej_wind[0]
    ), "log10_mej_wind should be -inf when mass_1 isn't a real NS under this EOS"
    assert not np.isfinite(log10_mej_total[0])


def test_kilonova_ejecta_fitting_requires_both_components_to_be_ns():
    """FIXME Weizmann: regression test for KilonovaEjectaFitting's routing.

    It used to route to bns_parameter_conversion whenever radius_1>0 alone,
    without also checking radius_2>0. mass_1 >= mass_2 by convention, so
    radius_1>0 usually implies radius_2>0 too (a lighter mass is inside the
    EOS's mass range whenever a heavier one is), but not always: mass_2
    can fall below the EOS table's tabulated minimum while mass_1 is a
    valid NS mass, and that row was still (wrongly) treated as BNS,
    silently publishing a finite ejecta mass instead of -inf (see
    test_bns_ejecta_conversion_rejects_non_ns_component for why the
    formula itself doesn't reliably catch this on its own).
    """
    # two rows, to force numpy's vectorized np.where routing path (the
    # if/elif scalar path only ever runs for single-injection calls, which
    # the real pipeline, always operating on a whole dataframe, never
    # does; a length-1 array can be truth-tested directly by numpy without
    # raising, so it wouldn't actually exercise the routing bug here)
    fitter = KilonovaEjectaFitting()
    params = dict(
        mass_1_source=np.array([1.8, 1.4]),
        mass_2_source=np.array([1.3, 1.3]),
        radius_1=np.array([13.0, 13.2]),  # row 0: >0, would wrongly route to BNS alone
        radius_2=np.array(
            [0.0, 13.1]
        ),  # row 0: <=0, mass_2 isn't a real NS; row 1: genuine BNS
        alpha=np.array([0.04, 0.04]),
        ratio_zeta=np.array([0.5, 0.5]),
        TOV_mass=np.array([2.0854, 2.0854]),
        R_16=np.array([12.0, 12.0]),
        # np.where evaluates both the bns_ and nsbh_parameter_conversion
        # branches eagerly for every row (only the result is masked
        # afterwards), so nsbh_parameter_conversion's inputs must be valid
        # for all rows too, even the ones that will end up routed to BNS.
        chi_1=np.array([0.0, 0.0]),
    )
    (
        log10_mej_dyn,
        log10_mej_wind,
        log10_mej_total,
        _,
    ) = fitter.ejecta_parameter_conversion(params)
    assert not np.isfinite(log10_mej_dyn[0]), "invalid secondary should give -inf"
    assert not np.isfinite(log10_mej_wind[0]), "invalid secondary should give -inf"
    assert not np.isfinite(log10_mej_total[0]), "invalid secondary should give -inf"
    assert np.isfinite(
        log10_mej_dyn[1]
    ), "a genuine BNS row should not be affected by the fix"


def test_binary_type_filter_end_to_end():
    """FIXME Weizmann: end-to-end regression test for --binary-type, which
    restores nmma 0.2.3's --binary-type/--eject behaviour: apply a single
    ejecta formula uniformly to every injection and drop (one-shot, no
    redraw) those whose resulting ejecta mass isn't finite. This is the
    only mechanism that can filter injections read from an external
    --gw-injection-file: --tests/Constraint-based filtering only works by
    redrawing samples, and a mass read from a file is fixed, so it can
    never be redrawn away from failing a test (see
    test_eos_injection_without_snr_test's module docstring era discussion;
    reproduced directly against the CLI here instead).

    Also checks that --binary-type overwrites log10_mej_dyn/log10_mej_wind
    even when the prior already samples them directly (nmma 0.2.3 had no
    "prefer the already-sampled value" behaviour, it always overwrote).
    """
    workingDir = os.path.dirname(__file__)
    dataDir = os.path.join(workingDir, "data")
    test_directory = os.path.join(dataDir, "binary_type_filter_test")
    exampleFilesDir = os.path.join(workingDir, "../../example_files/")
    if os.path.isdir(test_directory):
        shutil.rmtree(test_directory, ignore_errors=True)
    os.makedirs(test_directory, exist_ok=True)

    # Self-contained fixture (only alpha/ratio_zeta/ratio_epsilon, the
    # nuisance parameters bns_ejecta_conversion needs) rather than the
    # user's own priors/Bu2019lm_ejecta.prior: masses/luminosity_distance
    # come from gw_injection_path below, not the prior, for this test.
    prior_path = os.path.join(dataDir, "Bu2019lm_binary_type_test.prior")
    assert os.path.exists(prior_path), "prior file does not exist"

    # First 5 rows of the real, tracked example file: a known, fixed mix of
    # one BNS-under-ALF2, one NSBH-under-ALF2 and one BBH-under-ALF2 event
    # (masses verified by hand against ALF2's ~2.0854 Msun TOV mass).
    full_dat_path = os.path.join(exampleFilesDir, "sim_events", "bns_O4_injections.dat")
    assert os.path.exists(full_dat_path), "example gw-injection-file does not exist"
    gw_injection_path = os.path.join(test_directory, "bns_subset.dat")
    with open(full_dat_path) as f:
        lines = f.readlines()
    with open(gw_injection_path, "w") as f:
        f.writelines(lines[:6])  # header + first 5 rows

    injection_name = os.path.join(test_directory, "binary_type_injection.json")

    args = nmma_base_parsing(joint_parsing.injection_parsing, [])
    non_default_args = dict(
        prior_file=prior_path,
        gw_injection_file=gw_injection_path,
        injection_file=injection_name,
        eos_file="example_files/eos/ALF2.dat",
        original_parameters=True,
        generation_seed=42,
        binary_type="BNS",
    )
    for key, value in non_default_args.items():
        setattr(args, key, value)

    injection_handling.generate_injection(args)

    assert os.path.exists(injection_name), "injection file does not exist"
    injection_dict = read_injection_file(injection_name)
    n_kept = len(injection_dict)
    assert n_kept > 0, "--binary-type BNS filtered out every injection"
    assert n_kept < 5, (
        "--binary-type BNS should drop the non-BNS rows in this fixed "
        "5-row mix, not keep all of them (would indicate the prior's "
        "pre-sampled log10_mej_dyn/log10_mej_wind are being kept instead "
        "of being overwritten by the EOS-derived value)"
    )
    for radius_1, radius_2 in zip(
        injection_dict["radius_1"], injection_dict["radius_2"]
    ):
        assert radius_1 > 0 and radius_2 > 0, (
            "every kept injection must have both components be a real NS " "under ALF2"
        )

    shutil.rmtree(test_directory, ignore_errors=True)
