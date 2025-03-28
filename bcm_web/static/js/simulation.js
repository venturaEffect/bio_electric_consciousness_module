$(document).ready(function () {
  // Global variables
  let currentState = null;
  let iterationCount = 0;
  let autoRunInterval = null;
  let currentView = "voltage";
  let cellSize = 30; // Size of each cell in pixels for the click handler
  let timeSeriesData = {
    voltage: [],
    complexity: [],
  };

  // Initialize plots with dark theme
  initPlots();

  // Load scenarios and parameters
  loadScenarios();
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
    if (!currentState) return;

    // Get plot dimensions
    const plotElement = document.getElementById("visualization");
    const plotRect = plotElement.getBoundingClientRect();

    // Calculate the grid size from current state
    const gridSize = currentState.voltage_potential.length;

    // Calculate cell size in pixels
    const cellWidth = plotRect.width / gridSize;
    const cellHeight = plotRect.height / gridSize;

    // Calculate click position
    const x = Math.floor((e.clientX - plotRect.left) / cellWidth);
    const y = Math.floor((e.clientY - plotRect.top) / cellHeight);

    // Ensure coordinates are within bounds
    if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
      // Get currently selected ion and intensity
      const ionType = $("#ion-selector").val() || "voltage";
      const intensity = parseFloat($("#intensity-slider").val() || 0.8);

      console.log(
        `Modifying cell at (${x}, ${y}) with ${ionType} = ${intensity}`
      );

      // Send update request to server
      $.ajax({
        url: "/api/modify_cell",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
          x: x,
          y: y,
          ion: ionType,
          intensity: intensity,
          state: currentState,
        }),
        success: function (response) {
          if (response.success) {
            currentState = response.state;
            updateVisualization();
          } else {
            console.error("Error modifying cell:", response.error);
          }
        },
        error: function (xhr, status, error) {
          console.error("AJAX error:", error);
        },
      });
    }
  });

  // Add ion selector and intensity slider handlers
  $("#ion-selector").change(function () {
    // Update intensity slider label based on selected ion
    const ion = $(this).val();
    $("#intensity-label").text(
      `${ion.charAt(0).toUpperCase() + ion.slice(1)} Intensity`
    );
  });

  // Add experiment preset buttons
  $("#experiment-presets").change(function () {
    const experimentType = $(this).val();

    // Disable the control during experiment setup
    $(this).prop("disabled", true);

    // Request server to set up the experiment
    $.ajax({
      url: `/api/scenarios/${experimentType}`,
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify({
        steps: 50, // Initial run steps
      }),
      success: function (response) {
        if (response.success) {
          // Update UI with experiment state
          $("#experiment-description").text(response.description);
          currentState = response.state;
          iterationCount = response.iteration || 0;
          $("#iteration-count").val(iterationCount);
          updateVisualization();
        } else {
          console.error("Error setting up experiment:", response.error);
        }
        // Re-enable the control
        $("#experiment-presets").prop("disabled", false);
      },
      error: function (xhr, status, error) {
        console.error("AJAX error:", error);
        // Re-enable the control
        $("#experiment-presets").prop("disabled", false);
      },
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
              <label for="${param.section}-${param.name}" class="form-label text-light">${param.label}</label>
              <input type="range" class="form-range parameter-control" 
                  id="${param.section}-${param.name}"
                  data-section="${param.section}"
                  data-name="${param.name}"
                  min="${param.min}" max="${param.max}" step="${param.step}"
                  value="${param.default}">
              <div class="d-flex justify-content-between">
                  <span class="small text-light">${param.min}</span>
                  <span class="small value-display text-info">${param.default}</span>
                  <span class="small text-light">${param.max}</span>
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
    // Set dark theme for Plotly
    const darkTheme = {
      paper_bgcolor: "#222",
      plot_bgcolor: "#222",
      font: { color: "#fff" },
      title: { font: { color: "#fff" } },
      xaxis: {
        title: { font: { color: "#fff" } },
        color: "#fff",
        gridcolor: "#444",
      },
      yaxis: {
        title: { font: { color: "#fff" } },
        color: "#fff",
        gridcolor: "#444",
      },
    };

    // Create initial empty voltage potential plot
    const emptyHeatmap = {
      z: Array(10)
        .fill()
        .map(() => Array(10).fill(0)),
      type: "heatmap",
      colorscale: "Plasma",
    };

    Plotly.newPlot("visualization", [emptyHeatmap], {
      ...darkTheme,
      margin: { t: 0, l: 0, r: 0, b: 0 },
    });

    // Create empty time series plot
    const timeSeriesLayout = {
      ...darkTheme,
      title: "Average Voltage Over Time",
      xaxis: { ...darkTheme.xaxis, title: "Iteration" },
      yaxis: { ...darkTheme.yaxis, title: "Value" },
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
          line: { color: "#00ffff" }, // Cyan color for visibility on dark background
        },
      ],
      timeSeriesLayout
    );

    // Create empty pattern metrics plot
    const metricsLayout = {
      ...darkTheme,
      title: "Pattern Complexity",
      xaxis: { ...darkTheme.xaxis, title: "Iteration" },
      yaxis: { ...darkTheme.yaxis, title: "Complexity" },
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
          line: { color: "#ff9500" }, // Orange color for visibility on dark background
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
          colorscale: "Plasma",
          zmin: -1,
          zmax: 1,
          colorbar: {
            title: "Voltage (mV)",
            tickfont: { color: "#fff" },
            titlefont: { color: "#fff" },
          },
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
          colorscale: "YlOrBr",
          zmin: -1,
          zmax: 1,
          colorbar: {
            title: "Morphology",
            tickfont: { color: "#fff" },
            titlefont: { color: "#fff" },
          },
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
                ? "Bluered"
                : "Greens",
            zmin: -1,
            zmax: 1,
            colorbar: {
              title: `${
                ionName.charAt(0).toUpperCase() + ionName.slice(1)
              } (mM)`,
              tickfont: { color: "#fff" },
              titlefont: { color: "#fff" },
            },
          },
        ];
      }
    }

    // Update the plot with current data
    Plotly.react("visualization", data, {
      paper_bgcolor: "#222",
      plot_bgcolor: "#222",
      margin: { t: 0, l: 0, r: 0, b: 0 },
    });

    // Update time series data if we have state data
    if (currentState.voltage_potential) {
      const flatVoltage = currentState.voltage_potential.flat();
      const avgVoltage =
        flatVoltage.reduce((a, b) => a + b, 0) / flatVoltage.length;

      // Calculate a simple "complexity" metric - standard deviation of voltage
      const variance =
        flatVoltage.reduce((a, b) => a + Math.pow(b - avgVoltage, 2), 0) /
        flatVoltage.length;
      const complexity = Math.sqrt(variance);

      // Update time series
      Plotly.extendTraces(
        "time-series",
        {
          x: [[iterationCount]],
          y: [[avgVoltage]],
        },
        [0]
      );

      // Update metrics
      Plotly.extendTraces(
        "pattern-metrics",
        {
          x: [[iterationCount]],
          y: [[complexity]],
        },
        [0]
      );

      // Keep a limited window of data points
      const maxPoints = 100;
      if (iterationCount > maxPoints) {
        Plotly.relayout("time-series", {
          xaxis: {
            range: [iterationCount - maxPoints, iterationCount],
            color: "#fff",
            gridcolor: "#444",
          },
        });

        Plotly.relayout("pattern-metrics", {
          xaxis: {
            range: [iterationCount - maxPoints, iterationCount],
            color: "#fff",
            gridcolor: "#444",
          },
        });
      }
    }
  }

  function runSimulationStep() {
    // Collect updated parameters
    const configUpdates = {};

    $(".parameter-control").each(function () {
      const section = $(this).data("section");
      const name = $(this).data("name");
      const value = parseFloat($(this).val());

      if (!configUpdates[section]) {
        configUpdates[section] = {};
      }

      configUpdates[section][name] = value;
    });

    // Prepare request data
    const requestData = {
      config_updates: configUpdates,
      state: currentState,
      scenario: $("#scenario-select").val(),
    };

    // Send request to server
    $.ajax({
      url: "/api/run_step",
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify(requestData),
      success: function (response) {
        if (response.success) {
          currentState = response.state;
          iterationCount++;
          $("#iteration-count").val(iterationCount);

          // Update visualizations
          updateVisualization();
        } else {
          console.error("Error running simulation step:", response.error);
          stopAutoRun();
        }
      },
      error: function (xhr, status, error) {
        console.error("AJAX error:", error);
        stopAutoRun();
      },
    });
  }

  function resetSimulation() {
    // Stop auto-run if active
    stopAutoRun();

    // Reset state and counter
    currentState = null;
    iterationCount = 0;
    $("#iteration-count").val(0);

    // Reset time series data
    Plotly.react(
      "time-series",
      [
        {
          x: [],
          y: [],
          type: "scatter",
          mode: "lines",
          name: "Avg Voltage",
          line: { color: "#00ffff" },
        },
      ],
      {
        paper_bgcolor: "#222",
        plot_bgcolor: "#222",
        font: { color: "#fff" },
        title: {
          text: "Average Voltage Over Time",
          font: { color: "#fff" },
        },
        xaxis: {
          title: { text: "Iteration", font: { color: "#fff" } },
          color: "#fff",
          gridcolor: "#444",
        },
        yaxis: {
          title: { text: "Value", font: { color: "#fff" } },
          color: "#fff",
          gridcolor: "#444",
        },
        margin: { t: 30, l: 40, r: 10, b: 40 },
      }
    );

    Plotly.react(
      "pattern-metrics",
      [
        {
          x: [],
          y: [],
          type: "scatter",
          mode: "lines",
          name: "Complexity",
          line: { color: "#ff9500" },
        },
      ],
      {
        paper_bgcolor: "#222",
        plot_bgcolor: "#222",
        font: { color: "#fff" },
        title: {
          text: "Pattern Complexity",
          font: { color: "#fff" },
        },
        xaxis: {
          title: { text: "Iteration", font: { color: "#fff" } },
          color: "#fff",
          gridcolor: "#444",
        },
        yaxis: {
          title: { text: "Complexity", font: { color: "#fff" } },
          color: "#fff",
          gridcolor: "#444",
        },
        margin: { t: 30, l: 40, r: 10, b: 40 },
      }
    );

    // Run initial simulation step to get starting state
    runSimulationStep();

    // Reset button text
    $("#run-simulation").text("Run");
  }

  function startAutoRun() {
    // Clear existing interval if any
    stopAutoRun();

    // Start new interval
    autoRunInterval = setInterval(function () {
      runSimulationStep();
    }, 200); // Run every 200ms

    // Update checkbox
    $("#auto-run").prop("checked", true);
    $("#run-simulation").text("Stop");
  }

  function stopAutoRun() {
    if (autoRunInterval) {
      clearInterval(autoRunInterval);
      autoRunInterval = null;
    }

    // Update checkbox
    $("#auto-run").prop("checked", false);
    $("#run-simulation").text("Run");
  }

  function exportData() {
    if (!currentState) return;

    // Prepare data for export
    const exportData = {
      iteration: iterationCount,
      state: currentState,
      parameters: {},
    };

    // Add parameters
    $(".parameter-control").each(function () {
      const section = $(this).data("section");
      const name = $(this).data("name");
      const value = parseFloat($(this).val());

      if (!exportData.parameters[section]) {
        exportData.parameters[section] = {};
      }

      exportData.parameters[section][name] = value;
    });

    // Convert to JSON and create download link
    const dataStr =
      "data:text/json;charset=utf-8," +
      encodeURIComponent(JSON.stringify(exportData, null, 2));
    const downloadAnchorNode = document.createElement("a");
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute(
      "download",
      `bcm_simulation_${iterationCount}.json`
    );
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  }

  // Initialize by resetting
  resetSimulation();
});
