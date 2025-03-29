// Global variables
let currentState = null;
let currentView = "voltage";
let iterationCount = 0;
let simulationRunning = false;
let simulationInterval = null;
let isInitializing = false;

// Add this function at the beginning of your simulation.js file
function debugData(data, label) {
  console.log("-----DEBUG " + label + "-----");
  console.log(data);
  if (data && data.voltage_potential) {
    console.log(
      "Voltage range: " +
        Math.min(...data.voltage_potential.flat()) +
        " to " +
        Math.max(...data.voltage_potential.flat())
    );
  }
  console.log("-----------------------");
}

// Initialize application when DOM is ready
$(document).ready(function () {
  console.log("Initializing application");

  // Initialize plots with empty data
  initPlots();

  // Load parameters
  loadParameters();

  // Set up event handlers
  setupEventHandlers();

  // Run initial step after a short delay
  setTimeout(function () {
    initializeSimulation();
  }, 500);
});

function initPlots() {
  console.log("Initializing plots...");

  try {
    // Create empty grid data
    const emptyGrid = [
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
    ];

    // Main visualization - extremely simple to start
    Plotly.newPlot(
      "visualization",
      [
        {
          z: emptyGrid,
          type: "heatmap",
          colorscale: "Viridis",
        },
      ],
      {
        paper_bgcolor: "#222",
        plot_bgcolor: "#222",
        font: { color: "#fff" },
        margin: { t: 10, l: 50, r: 50, b: 10 },
        height: 350,
      }
    );

    // Time series plot - start with simple data
    Plotly.newPlot(
      "time-series",
      [
        {
          x: [0],
          y: [0.5],
          type: "scatter",
          mode: "lines",
          line: { color: "#00ff00" },
        },
      ],
      {
        paper_bgcolor: "#222",
        plot_bgcolor: "#222",
        font: { color: "#fff" },
        xaxis: { title: "Iteration", gridcolor: "#444", color: "#fff" },
        yaxis: { title: "Average Voltage", gridcolor: "#444", color: "#fff" },
        margin: { t: 10, l: 50, r: 20, b: 40 },
        height: 300,
      }
    );

    // Pattern complexity plot - start with simple data
    Plotly.newPlot(
      "pattern-metrics",
      [
        {
          x: [0],
          y: [0.2],
          type: "scatter",
          mode: "lines",
          line: { color: "#00ffff" },
        },
      ],
      {
        paper_bgcolor: "#222",
        plot_bgcolor: "#222",
        font: { color: "#fff" },
        xaxis: { title: "Iteration", gridcolor: "#444", color: "#fff" },
        yaxis: { title: "Complexity", gridcolor: "#444", color: "#fff" },
        margin: { t: 10, l: 50, r: 20, b: 40 },
        height: 300,
      }
    );

    console.log("Plots initialized");
  } catch (error) {
    console.error("Error initializing plots:", error);
  }
}

function loadParameters() {
  // Get parameters from server
  $.ajax({
    url: "/api/parameters",
    type: "GET",
    dataType: "json",
    success: function (data) {
      console.log("Parameters loaded:", data.length);
      populateParameters(data);
    },
    error: function (error) {
      console.error("Error loading parameters:", error);
      // Fallback to basic parameters if API fails
      const defaultParams = [
        {
          section: "core",
          name: "field_dimension",
          label: "Grid Size",
          min: 5,
          max: 20,
          step: 1,
          default: 10,
        },
        {
          section: "core",
          name: "gap_junction_strength",
          label: "Gap Junction Strength",
          min: 0,
          max: 1,
          step: 0.05,
          default: 0.5,
        },
        {
          section: "homeostasis",
          name: "homeostasis_strength",
          label: "Homeostasis",
          min: 0,
          max: 1,
          step: 0.05,
          default: 0.7,
        },
      ];
      populateParameters(defaultParams);
    },
  });
}

function populateParameters(parameters) {
  const container = $("#parameter-controls");
  container.empty();

  parameters.forEach((param) => {
    container.append(`
            <div class="mb-3">
                <label for="${param.section}-${param.name}" class="form-label">${param.label}</label>
                <input type="range" class="form-range parameter-slider" 
                       id="${param.section}-${param.name}" 
                       data-section="${param.section}"
                       data-name="${param.name}"
                       min="${param.min}" max="${param.max}" step="${param.step}" 
                       value="${param.default}">
                <div class="d-flex justify-content-between">
                    <small>${param.min}</small>
                    <small class="parameter-value">${param.default}</small>
                    <small>${param.max}</small>
                </div>
            </div>
        `);
  });

  // Update value display when slider changes
  $(".parameter-slider").on("input", function () {
    $(this).siblings("div").find(".parameter-value").text($(this).val());
  });
}

function setupEventHandlers() {
  // Run/pause button
  $("#run-btn").click(function () {
    if (simulationRunning) {
      stopSimulation();
    } else {
      startSimulation();
    }
  });

  // Step button
  $("#step-btn").click(function () {
    singleStep();
  });

  // Reset button
  $("#reset-btn").click(function () {
    resetSimulation();
  });

  // View selector buttons
  $('.btn-group[role="group"] .btn').click(function () {
    $('.btn-group[role="group"] .btn').removeClass("active");
    $(this).addClass("active");
    currentView = $(this).data("view");

    if (currentState) {
      updateVisualization();
    }
  });

  // Scenario selector
  $("#scenario-select").change(function () {
    if (!simulationRunning) {
      resetSimulation();
    }
  });

  // Add debug button handler
  $("#debug-btn").click(function () {
    const debugInfo = $("#debug-info");

    if (debugInfo.css("display") === "none") {
      debugInfo.css("display", "block");

      // Show current state info
      showDebugInfo({
        currentView,
        iterationCount,
        stateAvailable: currentState !== null,
        voltage: currentState
          ? {
              shape: [
                currentState.voltage_potential.length,
                currentState.voltage_potential[0].length,
              ],
              min: Math.min(...currentState.voltage_potential.flat()),
              max: Math.max(...currentState.voltage_potential.flat()),
            }
          : null,
        ions: currentState ? Object.keys(currentState.ion_gradients) : null,
      });
    } else {
      debugInfo.css("display", "none");
    }
  });
}

function initializeSimulation() {
  console.log("Initializing simulation...");
  if (isInitializing) return;

  isInitializing = true;

  // Ensure we create a good initial state with some pattern
  $.ajax({
    url: "/api/run_step",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      config_updates: {
        // Make sure we get a good initial pattern
        core: {
          field_dimension: 10,
          gap_junction_strength: 0.5,
        },
      },
      state: null, // Start fresh
      scenario: $("#scenario-select").val() || "default",
    }),
    success: function (response) {
      isInitializing = false;
      if (response.success) {
        currentState = response.state;
        debugData(currentState, "Initial State");
        updateVisualization();
        console.log("Initial state created");
      } else {
        console.error("Error initializing:", response.error || "Unknown error");
      }
    },
    error: function (xhr, status, error) {
      isInitializing = false;
      console.error("AJAX error during initialization:", error);
      console.log("Response:", xhr.responseText);
    },
  });
}

function startSimulation() {
  simulationRunning = true;
  $("#run-btn")
    .text("Pause")
    .removeClass("btn-outline-primary")
    .addClass("btn-outline-danger");

  // Run simulation loop at approximately 5 FPS
  simulationInterval = setInterval(singleStep, 200);
}

function stopSimulation() {
  simulationRunning = false;
  $("#run-btn")
    .text("Run")
    .removeClass("btn-outline-danger")
    .addClass("btn-outline-primary");

  if (simulationInterval) {
    clearInterval(simulationInterval);
    simulationInterval = null;
  }
}

function resetSimulation() {
  // First stop if running
  stopSimulation();

  // Reset iteration counter
  iterationCount = 0;
  $("#iteration-value").text("0");

  // Clear plots
  Plotly.animate("time-series", {
    data: [{ x: [0], y: [0] }],
  });

  Plotly.animate("pattern-metrics", {
    data: [{ x: [0], y: [0] }],
  });

  // Initialize with new scenario
  initializeSimulation();
}

function singleStep() {
  if (!currentState) {
    console.log("No current state, initializing first");
    initializeSimulation();
    return;
  }

  // Get parameter values from sliders
  let configUpdates = {};
  $(".parameter-slider").each(function () {
    const section = $(this).data("section");
    const name = $(this).data("name");
    const value = parseFloat($(this).val());

    if (!configUpdates[section]) {
      configUpdates[section] = {};
    }

    configUpdates[section][name] = value;
  });

  console.log("Running step with config:", configUpdates);

  // Run simulation step
  $.ajax({
    url: "/api/run_step",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      config_updates: configUpdates,
      state: currentState,
      scenario: $("#scenario-select").val(),
    }),
    success: function (response) {
      if (response.success) {
        console.log("Step response successful");
        currentState = response.state;
        iterationCount++;
        $("#iteration-value").text(iterationCount);
        updateVisualization();
      } else {
        console.error("Error during step:", response.error || "Unknown error");
        stopSimulation();
      }
    },
    error: function (xhr, status, error) {
      console.error("AJAX error during step:", error);
      console.log("Response:", xhr.responseText);
      stopSimulation();
    },
  });
}

// Replace your updateVisualization function with this version
function updateVisualization() {
  console.log("Updating visualization");

  if (!currentState) {
    console.error("No current state available");
    return;
  }

  debugData(currentState, "Current State");

  try {
    // Choose data based on current view
    let plotData = [];
    let colorScale = "Viridis";
    let title = "Voltage (mV)";

    if (currentView === "voltage" && currentState.voltage_potential) {
      // Check if data has proper structure
      if (
        !Array.isArray(currentState.voltage_potential) ||
        !Array.isArray(currentState.voltage_potential[0])
      ) {
        console.error("Voltage data is not a proper 2D array");
        debugData(currentState.voltage_potential, "Invalid voltage data");
        return;
      }

      // Log the actual values
      console.log(
        "Voltage data dimensions:",
        currentState.voltage_potential.length,
        currentState.voltage_potential[0].length
      );
      console.log(
        "Sample values:",
        currentState.voltage_potential[0][0],
        currentState.voltage_potential[0][1],
        currentState.voltage_potential[1][0]
      );

      plotData = [
        {
          z: currentState.voltage_potential,
          type: "heatmap",
          colorscale: colorScale,
          colorbar: {
            title: title,
            titlefont: { color: "#fff" },
            tickfont: { color: "#fff" },
          },
        },
      ];
    } else if (
      currentView === "sodium" &&
      currentState.ion_gradients &&
      currentState.ion_gradients.sodium
    ) {
      plotData = [
        {
          z: currentState.ion_gradients.sodium,
          type: "heatmap",
          colorscale: "Hot",
          colorbar: {
            title: "Sodium (mM)",
            titlefont: { color: "#fff" },
            tickfont: { color: "#fff" },
          },
        },
      ];
    } else if (
      currentView === "potassium" &&
      currentState.ion_gradients &&
      currentState.ion_gradients.potassium
    ) {
      plotData = [
        {
          z: currentState.ion_gradients.potassium,
          type: "heatmap",
          colorscale: "Blues",
          colorbar: {
            title: "Potassium (mM)",
            titlefont: { color: "#fff" },
            tickfont: { color: "#fff" },
          },
        },
      ];
    } else if (
      currentView === "calcium" &&
      currentState.ion_gradients &&
      currentState.ion_gradients.calcium
    ) {
      plotData = [
        {
          z: currentState.ion_gradients.calcium,
          type: "heatmap",
          colorscale: "Greens",
          colorbar: {
            title: "Calcium (mM)",
            titlefont: { color: "#fff" },
            tickfont: { color: "#fff" },
          },
        },
      ];
    } else if (
      currentView === "morphology" &&
      currentState.morphological_state
    ) {
      // Convert 1D array to 2D for visualization if needed
      if (Array.isArray(currentState.morphological_state[0])) {
        // Already 2D
        morphData = currentState.morphological_state;
      } else {
        // Convert 1D to 2D
        const size = Math.sqrt(currentState.morphological_state.length);
        morphData = [];
        for (let i = 0; i < size; i++) {
          let row = [];
          for (let j = 0; j < size; j++) {
            const idx = i * size + j;
            row.push(
              idx < currentState.morphological_state.length
                ? currentState.morphological_state[idx]
                : 0
            );
          }
          morphData.push(row);
        }
      }

      plotData = [
        {
          z: morphData,
          type: "heatmap",
          colorscale: "Portland",
          colorbar: {
            title: "Morphology",
            titlefont: { color: "#fff" },
            tickfont: { color: "#fff" },
          },
        },
      ];
    } else {
      console.error("No valid data for current view:", currentView);
      return;
    }

    console.log(
      "About to update plots with data:",
      plotData.length > 0 ? "valid data" : "no data"
    );

    // Update main visualization with specific layout
    const layout = {
      paper_bgcolor: "#222",
      plot_bgcolor: "#222",
      font: { color: "#fff" },
      margin: { t: 10, l: 50, r: 50, b: 10 },
      height: 350,
    };

    // Force redraw
    Plotly.purge("visualization");
    Plotly.newPlot("visualization", plotData, layout);

    // Update time series if we have voltage data
    if (currentState.voltage_potential) {
      // Calculate average voltage
      let sum = 0;
      let count = 0;

      for (let i = 0; i < currentState.voltage_potential.length; i++) {
        for (let j = 0; j < currentState.voltage_potential[i].length; j++) {
          sum += currentState.voltage_potential[i][j];
          count++;
        }
      }

      const avgVoltage = sum / count;

      // Calculate complexity (using std dev as a simple metric)
      let sumSqDiff = 0;
      for (let i = 0; i < currentState.voltage_potential.length; i++) {
        for (let j = 0; j < currentState.voltage_potential[i].length; j++) {
          sumSqDiff += Math.pow(
            currentState.voltage_potential[i][j] - avgVoltage,
            2
          );
        }
      }

      const complexity = Math.sqrt(sumSqDiff / count);

      // Update time series
      Plotly.extendTraces(
        "time-series",
        {
          x: [[iterationCount]],
          y: [[avgVoltage]],
        },
        [0]
      );

      // Update complexity metrics
      Plotly.extendTraces(
        "pattern-metrics",
        {
          x: [[iterationCount]],
          y: [[complexity]],
        },
        [0]
      );

      console.log("Updated time series with avg voltage:", avgVoltage);
      console.log("Updated complexity metrics with value:", complexity);
    }

    console.log("Visualization updated successfully");
  } catch (error) {
    console.error("Error updating visualization:", error);
  }
}
