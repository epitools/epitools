from magicgui.widgets import (
    CheckBox,
    Container,
    FloatSlider,
    Label,
    PushButton,
    Slider,
)


def projection() -> Container:
    """
    Widgets for epitools projection
    """

    smoothing_radius = FloatSlider(
        name="smoothing_radius", value=1.3, min=0.0, max=3.0
    )
    surface_smoothness_1 = Slider(
        name="surface_smoothness_1", value=30, min=0, max=100
    )
    surface_smoothness_2 = Slider(
        name="surface_smoothness_2", value=30, min=0, max=100
    )
    cut_off_distance = FloatSlider(
        name="cutoff_distance", value=1.2, min=0.0, max=3.0
    )
    run_button = PushButton(name="run_button", label="Run")

    projection_widget = Container(
        widgets=[
            smoothing_radius,
            surface_smoothness_1,
            surface_smoothness_2,
            cut_off_distance,
            run_button,
        ]
    )

    @projection_widget.run_button.clicked.connect
    def run() -> None:
        pass

    return projection_widget


def segmentation() -> Container:
    """
    Widgets for segmentation
    """
    seeding_label = Label(name="seeding_label", label="Seeding Cells")
    seed_gauss_smoothing = FloatSlider(
        name="gaussian_smoothing", value=1.0, min=0.0, max=4.0
    )
    min_cell_area = FloatSlider(
        name="minimum_cell_area", value=25.0, min=0.0, max=100.0
    )
    min_cell_intensity = FloatSlider(
        name="minimum_membrane_intensity", value=25.0, min=0.0, max=100.0
    )

    merging_label = Label(name="merging_label", label="Seed Merging")
    min_intensity_ratio = FloatSlider(
        name="min_intensity_ratio", value=0.35, min=0.0, max=1.0
    )

    segmentation_label = Label(name="segmentation_label", label="Segmentation")
    seg_gauss_smoothing = FloatSlider(
        name="seg_gauss_smoothing", value=2.0, min=0.0, max=5.0
    )

    false_positives_label = Label(
        name="false_positives_label", label="False Positives"
    )
    max_cell_area = Slider(
        name="max_cell_area", value=3000, min=1000, max=10000
    )
    min_mean_intensity = Slider(
        name="min_mean_intensity", value=30, min=0, max=100
    )

    use_clahe = CheckBox(name="use_clahe", value=False)

    test_button = PushButton(
        name="test_button", label="Test Segmentation (1st Frame)"
    )
    run_button = PushButton(
        name="run_button", label="Run Segmentation (all frames)"
    )

    segmentation_widget = Container(
        widgets=[
            seeding_label,
            seed_gauss_smoothing,
            min_cell_area,
            min_cell_intensity,
            merging_label,
            min_intensity_ratio,
            segmentation_label,
            seg_gauss_smoothing,
            false_positives_label,
            max_cell_area,
            min_mean_intensity,
            use_clahe,
            test_button,
            run_button,
        ]
    )

    @segmentation_widget.test_button.clicked.connect
    def test() -> None:
        pass

    @segmentation_widget.run_button.clicked.connect
    def run() -> None:
        pass

    return segmentation_widget
