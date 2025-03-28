$(document).ready(function () {
  // State variables
  let currentState = null;
  let iterationCount = 0;
  let autoRunInterval = null;
  let currentView = "voltage";
  let timeSeriesData = {
    voltage: [],
    complexity: [],
  };

  // Initialize plots
  initPlots();

  // Load scenarios
  loadScenarios();

  // Load parameters
  loadParameters();

  // Event handlers
  $("#scenario-select").change(function () {
    const scenarioId = $(this).val();
    updateScenarioDescription(scenarioId);
  });

  $("#reset-simulation").click(function () {
    resetSimulation();
  });

  $("#run-simulation").click(function () {
    if ($(this).text() === "Run") {
      startAutoRun();
      $(this).text("Stop");
    } else {
      stopAutoRun();
      $(this).text("Run");
    }
  });

  $("#step-button").click(function () {
    runSimulationStep();
  });

  $("#auto-run").change(function () {
    if ($(this).is(":checked")) {
      startAutoRun();
    } else {
      stopAutoRun();
    }
  });

  $('.btn-group[role="group"] button').click(function () {
    // Update active button
    $(this).siblings().removeClass("active");
    $(this).addClass("active");

    // Update current view
    currentView = $(this).data("view");

    // Update visualization
    if (currentState) {
      updateVisualization();
    }
  });

  $("#export-data").click(function () {
    exportData();
  });

  // Add interactive cell state modification
  $("#visualization").on("click", function (e) {
    // Convert click coordinates to cell position
    const rect = this.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / cellSize);
    const y = Math.floor((e.clientY - rect.top) / cellSize);

    // Get currently selected ion and intensity
    const ionType = $("#ion-selector").val();
    const intensity = $("#intensity-slider").val();

    // Send update request to server
    $.post("/api/modify_cell", {
      x: x,
      y: y,
      ion: ionType,
      intensity: intensity,
    }).done(function (response) {
      // Update visualization with new state
      updateVisualization(response.state);
    });
  });

  // Add experiment preset buttons
  $("#experiment-presets").on("change", function () {
    const experimentType = $(this).val();

    // Request server to set up the experiment
    $.post(`/api/scenarios/${experimentType}`, {
      steps: 50, // Initial run steps
    }).done(function (response) {
      // Update UI with experiment state
      $("#experiment-description").text(response.description);
      updateVisualization(response.state);
    });
  });

  // Functions
  function loadScenarios() {
    $.get("/api/scenarios", function (scenarios) {
      const select = $("#scenario-select");
      select.empty();

      scenarios.forEach(function (scenario) {
        select.append(
          `<option value="${scenario.id}">${scenario.name}</option>`
        );
      });

      // Set initial description
      updateScenarioDescription(scenarios[0].id);
    });
  }

  function updateScenarioDescription(scenarioId) {
    $.get("/api/scenarios", function (scenarios) {
      const scenario = scenarios.find((s) => s.id === scenarioId);
      $("#scenario-description").text(scenario.description);
    });
  }

  function loadParameters() {
    $.get("/api/parameters", function (parameters) {
      const container = $("#parameters-container");
      container.empty();

      parameters.forEach(function (param) {
        const html = `
                    <div class="mb-3">
                        <label for="${param.section}-${param.name}" class="form-label">${param.label}</label>
                        <input type="range" class="form-range parameter-control" 
                            id="${param.section}-${param.name}"
                            data-section="${param.section}"
                            data-name="${param.name}"
                            min="${param.min}" max="${param.max}" step="${param.step}"
                            value="${param.default}">
                        <div class="d-flex justify-content-between">
                            <span class="small">${param.min}</span>
                            <span class="small value-display">${param.default}</span>
                            <span class="small">${param.max}</span>
                        </div>
                        <p class="text-muted small">${param.description}</p>
                    </div>
                `;
        container.append(html);
      });

      // Add event listeners for sliders
      $(".parameter-control").on("input", function () {
        // Update display value
        $(this).closest(".mb-3").find(".value-display").text($(this).val());
      });
    });
  }

  function initPlots() {
    // Create initial empty voltage potential plot
    const emptyHeatmap = {
      z: Array(10)
        .fill()
        .map(() => Array(10).fill(0)),
      type: "heatmap",
      colorscale: "Viridis",
    };

    Plotly.newPlot("visualization", [emptyHeatmap]);

    // Create empty time series plot
    const timeSeriesLayout = {
      title: "Average Voltage Over Time",
      xaxis: { title: "Iteration" },
      yaxis: { title: "Value" },
      margin: { t: 30, l: 40, r: 10, b: 40 },
    };

    Plotly.newPlot(
      "time-series",
      [
        {
          x: [],
          y: [],
          type: "scatter",
          mode: "lines",
          name: "Avg Voltage",
        },
      ],
      timeSeriesLayout
    );

    // Create empty pattern metrics plot
    const metricsLayout = {
      title: "Pattern Complexity",
      xaxis: { title: "Iteration" },
      yaxis: { title: "Complexity" },
      margin: { t: 30, l: 40, r: 10, b: 40 },
    };

    Plotly.newPlot(
      "pattern-metrics",
      [
        {
          x: [],
          y: [],
          type: "scatter",
          mode: "lines",
          name: "Complexity",
          line: { color: "orange" },
        },
      ],
      metricsLayout
    );
  }

  function updateVisualization() {
    let data = [];

    if (currentView === "voltage") {
      data = [
        {
          z: currentState.voltage_potential,
          type: "heatmap",
          colorscale: "Viridis",
          zmin: -1,
          zmax: 1,
          colorbar: { title: "Voltage (mV)" },
        },
      ];
    } else if (currentView === "morphology") {
      // Reshape 1D morphological state to 2D for visualization
      const size = Math.sqrt(currentState.morphological_state.length);
      const morphology2D = [];

      for (let i = 0; i < size; i++) {
        const row = [];
        for (let j = 0; j < size; j++) {
          row.push(currentState.morphological_state[i * size + j]);
        }
        morphology2D.push(row);
      }

      data = [
        {
          z: morphology2D,
          type: "heatmap",
          colorscale: "Cividis",
          zmin: -1,
          zmax: 1,
          colorbar: { title: "Morphology" },
        },
      ];
    } else {
      // Ion gradient views (sodium, potassium, calcium)
      const ionName = currentView;

      if (currentState.ion_gradients[ionName]) {
        data = [
          {
            z: currentState.ion_gradients[ionName],
            type: "heatmap",
            colorscale:
              ionName === "sodium"
                ? "Hot"
                : ionName === "potassium"
                ? "Blues"
                : "Greens",
            zmin: -1,
            zmax: 1,
            colorbar: {
              title: `${
                ionName.charAt(0).toUpperCase() + ionName.slice(1)
              } (mM)`,
            },
          },
        ];
      }
    }
  }
});
