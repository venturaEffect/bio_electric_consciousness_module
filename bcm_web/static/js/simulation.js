// Global variables
let currentState = null;
let iterationCount = 0;
let autoRunInterval = null;
let currentView = "voltage";

// Initialize when the document is ready
$(document).ready(function () {
  console.log("Document ready, initializing application...");

  // Initialize plots with empty data
  initPlots();

  // Load scenarios and parameters
  loadScenarios();
  loadParameters();

  // Set up event handlers
  setupEventHandlers();

  // Reset simulation to initialize with default state
  resetSimulation();

  console.log("Initialization completed");
});

// Initialize visualization plots
function initPlots() {
  console.log("Initializing plots...");

  // Create initial empty grid for visualization
  const size = 10; // Default size
  const emptyGrid = Array(size)
    .fill()
    .map(() => Array(size).fill(0));

  // Main visualization
  Plotly.newPlot(
    "visualization",
    [
      {
        z: emptyGrid,
        type: "heatmap",
        colorscale: "Viridis",
        showscale: true,
        colorbar: {
          title: "Voltage (mV)",
          titlefont: { color: "#fff" },
          tickfont: { color: "#fff" },
        },
      },
    ],
    {
      paper_bgcolor: "#222",
      plot_bgcolor: "#222",
      font: { color: "#fff" },
      margin: { t: 10, l: 50, r: 50, b: 10 },
    }
  );

  // Time series plot
  Plotly.newPlot(
    "time-series",
    [
      {
        x: [],
        y: [],
        type: "scatter",
        mode: "lines",
        name: "Avg Voltage",
        line: { color: "#00ff00" },
      },
    ],
    {
      paper_bgcolor: "#222",
      plot_bgcolor: "#222",
      font: { color: "#fff" },
      title: { text: "Voltage Potential Over Time", font: { color: "#fff" } },
      xaxis: { title: "Iterations", color: "#fff", gridcolor: "#444" },
      yaxis: { title: "Average Voltage", color: "#fff", gridcolor: "#444" },
      margin: { t: 30, l: 60, r: 10, b: 40 },
    }
  );

  // Pattern metrics
  Plotly.newPlot(
    "pattern-metrics",
    [
      {
        x: [],
        y: [],
        type: "scatter",
        mode: "lines",
        name: "Complexity",
        line: { color: "#00ffff" },
      },
    ],
    {
      paper_bgcolor: "#222",
      plot_bgcolor: "#222",
      font: { color: "#fff" },
      title: { text: "Pattern Complexity", font: { color: "#fff" } },
      xaxis: { title: "Iterations", color: "#fff", gridcolor: "#444" },
      yaxis: { title: "Complexity", color: "#fff", gridcolor: "#444" },
      margin: { t: 30, l: 60, r: 10, b: 40 },
    }
  );

  console.log("Plots initialized successfully");
}

// Setup all event handlers
function setupEventHandlers() {
  // View selector buttons
  $(".btn-group button").on("click", function () {
    $(".btn-group button").removeClass("active");
    $(this).addClass("active");
    currentView = $(this).data("view");
    updateVisualization();
  });

  // Reset button
  $("#reset-simulation").on("click", function () {
    resetSimulation();
  });

  // Run/stop button
  $("#run-simulation").on("click", function () {
    if ($(this).text() === "Run") {
      startAutoRun();
      $(this).text("Stop");
    } else {
      stopAutoRun();
      $(this).text("Run");
    }
  });

  // Step button
  $("#step-button").on("click", function () {
    runSimulationStep();
  });

  // Auto-run checkbox
  $("#auto-run").on("change", function () {
    if ($(this).is(":checked")) {
      startAutoRun();
      $("#run-simulation").text("Stop");
    } else {
      stopAutoRun();
      $("#run-simulation").text("Run");
    }
  });

  // Scenario selector
  $("#scenario-select").on("change", function () {
    resetSimulation();
  });

  // Intensity slider
  $("#intensity-slider").on("input", function () {
    $("#intensity-value").text($(this).val());
  });

  // Visualization click for injection
  $("#visualization").on("click", function (event) {
    // Get click coordinates and convert to grid position
    const container = document.getElementById("visualization");
    const rect = container.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Grid size
    const size = currentState.voltage_potential.length;
    const cellWidth = rect.width / size;
    const cellHeight = rect.height / size;

    // Calculate grid indices
    const col = Math.floor(x / cellWidth);
    const row = Math.floor(y / cellHeight);

    // Ensure we're within bounds
    if (row >= 0 && row < size && col >= 0 && col < size) {
      // Get selected ion and intensity
      const selectedIon = $("#ion-selector").val();
      const intensity = parseFloat($("#intensity-slider").val());

      // Call API to modify cell
      modifyCell(row, col, selectedIon, intensity);
    }
  });

  // Export data button
  $("#export-data").on("click", function () {
    exportData();
  });
}

// Load available scenarios from API
function loadScenarios() {
  $.ajax({
    url: "/api/scenarios",
    type: "GET",
    success: function (data) {
      const select = $("#scenario-select");
      select.empty();

      data.scenarios.forEach(function (scenario) {
        select.append(
          $("<option></option>").attr("value", scenario.id).text(scenario.name)
        );

        // Set description for first scenario
        if (select.val() === scenario.id) {
          $("#scenario-description").text(scenario.description);
        }
      });

      // Update description when selection changes
      select.on("change", function () {
        const scenarioId = $(this).val();
        const scenario = data.scenarios.find((s) => s.id === scenarioId);
        $("#scenario-description").text(scenario ? scenario.description : "");
      });
    },
    error: function (xhr, status, error) {
      console.error("Error loading scenarios:", error);
      $("#scenario-description").text("Failed to load scenarios from server.");
    },
  });
}

// Load parameters from API
function loadParameters() {
  $.ajax({
    url: "/api/parameters",
    type: "GET",
    success: function (data) {
      const container = $("#parameters-container");
      container.empty();

      // Create accordion for parameter sections
      const accordion = $(
        '<div class="accordion" id="parameters-accordion"></div>'
      );
      container.append(accordion);

      // Process each section
      Object.keys(data.parameters).forEach(function (section, index) {
        const params = data.parameters[section];
        const sectionId = `section-${section
          .replace(/\s+/g, "-")
          .toLowerCase()}`;

        // Create accordion item
        const accordionItem = $(`
                    <div class="accordion-item bg-dark border-secondary">
                        <h2 class="accordion-header">
                            <button class="accordion-button bg-dark text-light collapsed" type="button" 
                                    data-bs-toggle="collapse" data-bs-target="#${sectionId}">
                                ${section}
                            </button>
                        </h2>
                        <div id="${sectionId}" class="accordion-collapse collapse" 
                             data-bs-parent="#parameters-accordion">
                            <div class="accordion-body"></div>
                        </div>
                    </div>
                `);

        accordion.append(accordionItem);

        // Add parameters to this section
        const body = accordionItem.find(".accordion-body");

        Object.keys(params).forEach(function (name) {
          const param = params[name];
          const inputId = `param-${section}-${name}`
            .replace(/\s+/g, "-")
            .toLowerCase();

          const formGroup = $(`
                        <div class="mb-3">
                            <label for="${inputId}" class="form-label">${
            param.display_name || name
          }</label>
                            <input type="range" class="form-range parameter-control" 
                                   id="${inputId}" min="${param.min}" max="${
            param.max
          }" 
                                   step="${param.step}" value="${param.default}"
                                   data-section="${section}" data-name="${name}">
                            <div class="d-flex justify-content-between">
                                <span class="small">${param.min}</span>
                                <span class="small param-value">${
                                  param.default
                                }</span>
                                <span class="small">${param.max}</span>
                            </div>
                        </div>
                    `);

          body.append(formGroup);

          // Update displayed value when slider changes
          formGroup.find(".form-range").on("input", function () {
            formGroup.find(".param-value").text($(this).val());
          });
        });
      });

      // Open first section by default
      accordion.find(".accordion-collapse").first().addClass("show");
      accordion.find(".accordion-button").first().removeClass("collapsed");
    },
    error: function (xhr, status, error) {
      console.error("Error loading parameters:", error);
      $("#parameters-container").html(
        '<p class="text-danger">Failed to load parameters.</p>'
      );
    },
  });
}

// Update the visualization with current state data
function updateVisualization() {
  console.log("Updating visualization with current state:", currentState);

  if (!currentState || !currentState.voltage_potential) {
    console.error("Invalid state data for visualization");
    return;
  }

  let plotData = [];
  let colorScale = "Viridis";
  let title = "Voltage (mV)";

  // Select data and appearance based on current view
  switch (currentView) {
    case "voltage":
      plotData = currentState.voltage_potential;
      colorScale = "Viridis";
      title = "Voltage (mV)";
      break;

    case "sodium":
      plotData = currentState.ion_gradients.sodium;
      colorScale = "Hot";
      title = "Sodium (mM)";
      break;

    case "potassium":
      plotData = currentState.ion_gradients.potassium;
      colorScale = "Blues";
      title = "Potassium (mM)";
      break;

    case "calcium":
      plotData = currentState.ion_gradients.calcium;
      colorScale = "Greens";
      title = "Calcium (mM)";
      break;

    case "morphology":
      // Reshape morphological state to 2D grid
      const size = Math.sqrt(currentState.morphological_state.length);
      plotData = [];
      for (let i = 0; i < size; i++) {
        const row = [];
        for (let j = 0; j < size; j++) {
          row.push(currentState.morphological_state[i * size + j]);
        }
        plotData.push(row);
      }
      colorScale = "Portland";
      title = "Morphological State";
      break;
  }

  // Update the main visualization
  Plotly.react(
    "visualization",
    [
      {
        z: plotData,
        type: "heatmap",
        colorscale: colorScale,
        showscale: true,
        colorbar: {
          title: title,
          titlefont: { color: "#fff" },
          tickfont: { color: "#fff" },
        },
      },
    ],
    {
      paper_bgcolor: "#222",
      plot_bgcolor: "#222",
      font: { color: "#fff" },
      margin: { t: 10, l: 50, r: 50, b: 10 },
    }
  );

  // Update time series charts if we have voltage data
  if (currentState.voltage_potential) {
    // Calculate average voltage
    let sum = 0;
    let count = 0;
    currentState.voltage_potential.forEach((row) => {
      row.forEach((val) => {
        sum += val;
        count++;
      });
    });
    const avgVoltage = sum / count;

    // Calculate complexity (using standard deviation as a simple metric)
    let sumSqDiff = 0;
    currentState.voltage_potential.forEach((row) => {
      row.forEach((val) => {
        sumSqDiff += Math.pow(val - avgVoltage, 2);
      });
    });
    const complexity = Math.sqrt(sumSqDiff / count);

    // Update time series plot
    Plotly.extendTraces(
      "time-series",
      {
        x: [[iterationCount]],
        y: [[avgVoltage]],
      },
      [0]
    );

    // Update complexity plot
    Plotly.extendTraces(
      "pattern-metrics",
      {
        x: [[iterationCount]],
        y: [[complexity]],
      },
      [0]
    );

    // Keep a reasonable window of data points visible
    const maxDataPoints = 100;
    if (iterationCount > maxDataPoints) {
      Plotly.relayout("time-series", {
        xaxis: {
          range: [iterationCount - maxDataPoints, iterationCount],
          color: "#fff",
          gridcolor: "#444",
        },
      });

      Plotly.relayout("pattern-metrics", {
        xaxis: {
          range: [iterationCount - maxDataPoints, iterationCount],
          color: "#fff",
          gridcolor: "#444",
        },
      });
    }
  }
}

// Run one simulation step
function runSimulationStep() {
  // Collect current parameter values
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

  // Build request data
  const requestData = {
    config_updates: configUpdates,
    state: currentState,
    scenario: $("#scenario-select").val(),
  };

  console.log("Sending request data:", requestData);

  // Send to server
  $.ajax({
    url: "/api/run_step",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify(requestData),
    success: function (response) {
      console.log("Received response:", response);

      if (response.success) {
        // Update state and iteration count
        currentState = response.state;
        iterationCount++;
        $("#iteration-count").val(iterationCount);

        // Update visualization
        updateVisualization();
      } else {
        console.error("Error running simulation step:", response.error);
        if (autoRunInterval) {
          stopAutoRun();
        }
        alert("Error: " + response.error);
      }
    },
    error: function (xhr, status, error) {
      console.error("AJAX error:", xhr.status, error);
      if (autoRunInterval) {
        stopAutoRun();
      }
      alert("Network error: " + error);
    },
  });
}

// Start auto-running the simulation
function startAutoRun() {
  if (!autoRunInterval) {
    autoRunInterval = setInterval(runSimulationStep, 200);
    $("#auto-run").prop("checked", true);
    $("#run-simulation").text("Stop");
  }
}

// Stop auto-running the simulation
function stopAutoRun() {
  if (autoRunInterval) {
    clearInterval(autoRunInterval);
    autoRunInterval = null;
    $("#auto-run").prop("checked", false);
    $("#run-simulation").text("Run");
  }
}

// Reset simulation to initial state
function resetSimulation() {
  console.log("Resetting simulation");

  // Stop auto-run if active
  stopAutoRun();

  // Reset state and iteration counter
  currentState = null;
  iterationCount = 0;
  $("#iteration-count").val(0);

  // Clear charts
  Plotly.react(
    "time-series",
    [
      {
        x: [],
        y: [],
        type: "scatter",
        mode: "lines",
        name: "Avg Voltage",
        line: { color: "#00ff00" },
      },
    ],
    {
      paper_bgcolor: "#222",
      plot_bgcolor: "#222",
      font: { color: "#fff" },
      title: { text: "Voltage Potential Over Time", font: { color: "#fff" } },
      xaxis: { title: "Iterations", color: "#fff", gridcolor: "#444" },
      yaxis: { title: "Average Voltage", color: "#fff", gridcolor: "#444" },
      margin: { t: 30, l: 60, r: 10, b: 40 },
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
        line: { color: "#00ffff" },
      },
    ],
    {
      paper_bgcolor: "#222",
      plot_bgcolor: "#222",
      font: { color: "#fff" },
      title: { text: "Pattern Complexity", font: { color: "#fff" } },
      xaxis: { title: "Iterations", color: "#fff", gridcolor: "#444" },
      yaxis: { title: "Complexity", color: "#fff", gridcolor: "#444" },
      margin: { t: 30, l: 60, r: 10, b: 40 },
    }
  );

  // Run initial simulation step to get starting state
  runSimulationStep();
}

// Modify a cell in the grid
function modifyCell(row, col, type, intensity) {
  if (!currentState) {
    console.error("No current state to modify");
    return;
  }

  $.ajax({
    url: "/api/modify_cell",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      row: row,
      col: col,
      type: type,
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
      console.error("AJAX error during cell modification:", error);
    },
  });
}

// Export current simulation data
function exportData() {
  if (!currentState) {
    alert("No simulation data to export.");
    return;
  }

  const dataStr = JSON.stringify(
    {
      state: currentState,
      iteration: iterationCount,
      timestamp: new Date().toISOString(),
      parameters: collectCurrentParameters(),
    },
    null,
    2
  );

  // Create download link
  const dataBlob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(dataBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `bioelectric-sim-${new Date()
    .toISOString()
    .slice(0, 19)
    .replace(/:/g, "-")}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Collect all current parameter values
function collectCurrentParameters() {
  const params = {};

  $(".parameter-control").each(function () {
    const section = $(this).data("section");
    const name = $(this).data("name");
    const value = parseFloat($(this).val());

    if (!params[section]) {
      params[section] = {};
    }

    params[section][name] = value;
  });

  return params;
}
