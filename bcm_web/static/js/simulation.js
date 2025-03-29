// Global variables
let currentState = null;
let currentView = "voltage";
let iterationCount = 0;
let simulationRunning = false;
let simulationInterval = null;

// Initialize application when DOM is ready
$(document).ready(function () {
  console.log("Initializing application");

  // Initialize plots
  initPlots();

  // Load scenarios
  loadScenarios();

  // Load parameters
  loadParameters();

  // Set up event handlers
  setupEventHandlers();

  // Run initial step to get started
  setTimeout(function () {
    runInitialStep();
  }, 1000);
});

function runInitialStep() {
  // Send request to initialize simulation
  $.ajax({
    url: "/api/run_step",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      config_updates: {},
      state: null,
      scenario: "default",
    }),
    success: function (response) {
      console.log("Initial step response:", response);
      if (response.success) {
        currentState = response.state;
        updateVisualization();
        console.log("Initial visualization complete");
      } else {
        console.error("Error in initial step:", response.error);
      }
    },
    error: function (xhr, status, error) {
      console.error("AJAX error:", status, error);
    },
  });
}

function initPlots() {
  console.log("Initializing plots...");

  try {
    // Initialize main visualization with dummy data
    const dummyData = Array(10)
      .fill()
      .map(() => Array(10).fill(0));

    Plotly.newPlot(
      "visualization",
      [
        {
          z: dummyData,
          type: "heatmap",
          colorscale: "Viridis",
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
        height: 300, // Explicitly set height
      }
    );

    // Initialize time series plot
    Plotly.newPlot(
      "time-series",
      [
        {
          x: [],
          y: [],
          type: "scatter",
          mode: "lines",
          line: { color: "#00ff00", width: 2 },
          name: "Average Voltage",
        },
      ],
      {
        paper_bgcolor: "#222",
        plot_bgcolor: "#222",
        font: { color: "#fff" },
        margin: { t: 10, l: 50, r: 20, b: 40 },
        xaxis: { title: "Iteration", gridcolor: "#444", color: "#fff" },
        yaxis: { title: "Voltage", gridcolor: "#444", color: "#fff" },
        height: 300, // Explicitly set height
      }
    );

    // Initialize pattern metrics plot
    Plotly.newPlot(
      "pattern-metrics",
      [
        {
          x: [],
          y: [],
          type: "scatter",
          mode: "lines",
          line: { color: "#00ffff", width: 2 },
          name: "Complexity",
        },
      ],
      {
        paper_bgcolor: "#222",
        plot_bgcolor: "#222",
        font: { color: "#fff" },
        margin: { t: 10, l: 50, r: 20, b: 40 },
        xaxis: { title: "Iteration", gridcolor: "#444", color: "#fff" },
        yaxis: { title: "Complexity", gridcolor: "#444", color: "#fff" },
        height: 300, // Explicitly set height
      }
    );

    console.log("Plots initialized successfully");
  } catch (error) {
    console.error("Error initializing plots:", error);
  }
}

function loadScenarios() {
  $.get("/api/scenarios", function (data) {
    const select = $("#scenario-select");
    select.empty();

    data.forEach((scenario) => {
      select.append(`<option value="${scenario.id}">${scenario.name}</option>`);
    });

    console.log("Scenarios loaded:", data.length);
  });
}

function loadParameters() {
  $.get("/api/parameters", function (data) {
    const container = $("#parameter-controls");
    container.empty();

    data.forEach((param) => {
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

    console.log("Parameters loaded:", data.length);
  });
}

function setupEventHandlers() {
  // Start button
  $("#start-btn").click(function () {
    if (!simulationRunning) {
      startSimulation();
    } else {
      pauseSimulation();
    }
  });

  // Step button
  $("#step-btn").click(function () {
    runStep();
  });

  // Reset button
  $("#reset-btn").click(function () {
    resetSimulation();
  });

  // View selector
  $('.btn-group[role="group"] .btn').click(function () {
    $('.btn-group[role="group"] .btn').removeClass("active");
    $(this).addClass("active");
    currentView = $(this).data("view");

    if (currentState) {
      updateVisualization();
    }
  });

  // Click on visualization to modify cell
  $("#visualization").on("click", function (e) {
    if (!currentState) return;

    const rect = e.target.getBoundingClientRect();
    const offsetX = e.clientX - rect.left;
    const offsetY = e.clientY - rect.top;

    // Convert click coordinates to cell grid coordinates
    const width = $(this).width();
    const height = $(this).height();
    const gridSize = currentState.voltage_potential.length;

    const x = Math.floor((offsetX / width) * gridSize);
    const y = Math.floor((offsetY / height) * gridSize);

    // Get sliders values
    const intensity = parseFloat($("#homeostasis-homeostasis_strength").val());

    // Modify cell
    $.ajax({
      url: "/api/modify_cell",
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify({
        x: x,
        y: y,
        ion: currentView === "voltage" ? "voltage" : currentView,
        intensity: intensity,
      }),
      success: function (response) {
        if (response.success) {
          currentState = response.state;
          updateVisualization();
        }
      },
    });
  });
}

function startSimulation() {
  simulationRunning = true;
  $("#start-btn")
    .text("Pause")
    .removeClass("btn-outline-primary")
    .addClass("btn-outline-warning");

  // Run simulation at approximately 10 FPS
  simulationInterval = setInterval(function () {
    runStep();
  }, 100);
}

function pauseSimulation() {
  simulationRunning = false;
  $("#start-btn")
    .text("Start")
    .removeClass("btn-outline-warning")
    .addClass("btn-outline-primary");

  if (simulationInterval) {
    clearInterval(simulationInterval);
  }
}

function resetSimulation() {
  pauseSimulation();
  iterationCount = 0;
  $("#iteration-counter").text("Iteration: 0");

  // Get selected scenario
  const scenario = $("#scenario-select").val();

  // Initialize simulation with selected scenario
  $.ajax({
    url: "/api/run_step",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      config_updates: {},
      state: null,
      scenario: scenario,
    }),
    success: function (response) {
      console.log("Reset response:", response);
      if (response.success) {
        currentState = response.state;
        updateVisualization();
      } else {
        console.error("Error resetting simulation:", response.error);
      }
    },
    error: function (xhr, status, error) {
      console.error("AJAX error during reset:", status, error);
    },
  });
}

function runStep() {
  // Get current parameter values
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

  // Get selected scenario
  const scenario = $("#scenario-select").val();

  // Send request to run a step
  $.ajax({
    url: "/api/run_step",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      config_updates: configUpdates,
      state: currentState,
      scenario: scenario,
    }),
    success: function (response) {
      if (response.success) {
        // Update state
        currentState = response.state;

        // Update iteration count
        iterationCount++;
        $("#iteration-counter").text("Iteration: " + iterationCount);

        // Update visualization
        updateVisualization();
      } else {
        console.error("Error running step:", response.error || "Unknown error");
        pauseSimulation();
      }
    },
    error: function (xhr, status, error) {
      console.error("AJAX error during step:", status, error);
      pauseSimulation();
    },
  });
}

function updateVisualization() {
  console.log("Updating visualization with state:", currentState);

  if (!currentState) {
    console.error("No current state available");
    return;
  }

  // Add debug information to help troubleshoot
  console.log("Current view:", currentView);
  console.log(
    "Voltage potential shape:",
    currentState.voltage_potential
      ? [
          currentState.voltage_potential.length,
          currentState.voltage_potential[0].length,
        ]
      : "undefined"
  );

  // Select data based on current view
  let plotData = [];

  try {
    if (currentView === "voltage" && currentState.voltage_potential) {
      plotData = [
        {
          z: currentState.voltage_potential,
          type: "heatmap",
          colorscale: "Viridis",
          colorbar: {
            title: "Voltage (mV)",
            titlefont: { color: "#fff" },
            tickfont: { color: "#fff" },
          },
        },
      ];
    } else if (
      currentView === "morphology" &&
      currentState.morphological_state
    ) {
      // Convert 1D morphological state to 2D for visualization
      const morphState = currentState.morphological_state;
      const size = Math.floor(Math.sqrt(morphState.length));
      let morphGrid = [];

      for (let i = 0; i < size; i++) {
        let row = [];
        for (let j = 0; j < size; j++) {
          const idx = i * size + j;
          row.push(idx < morphState.length ? morphState[idx] : 0);
        }
        morphGrid.push(row);
      }

      plotData = [
        {
          z: morphGrid,
          type: "heatmap",
          colorscale: "Cividis",
          colorbar: {
            title: "Morphological State",
            titlefont: { color: "#fff" },
            tickfont: { color: "#fff" },
          },
        },
      ];
    } else if (
      currentState.ion_gradients &&
      currentState.ion_gradients[currentView]
    ) {
      // Ion gradients (sodium, potassium, calcium)
      let colorscale = "Hot";
      if (currentView === "potassium") colorscale = "Blues";
      if (currentView === "calcium") colorscale = "Greens";

      plotData = [
        {
          z: currentState.ion_gradients[currentView],
          type: "heatmap",
          colorscale: colorscale,
          colorbar: {
            title: `${
              currentView.charAt(0).toUpperCase() + currentView.slice(1)
            } (mM)`,
            titlefont: { color: "#fff" },
            tickfont: { color: "#fff" },
          },
        },
      ];
    }

    // Update the main visualization with updated layout
    Plotly.react("visualization", plotData, {
      paper_bgcolor: "#222",
      plot_bgcolor: "#222",
      font: { color: "#fff" },
      margin: { t: 10, l: 50, r: 50, b: 10 },
      height: 300, // Explicitly set height
    });

    // Update time series plot if we have state data
    if (currentState.voltage_potential) {
      // Calculate average voltage
      let totalVoltage = 0;
      let count = 0;

      for (let i = 0; i < currentState.voltage_potential.length; i++) {
        for (let j = 0; j < currentState.voltage_potential[i].length; j++) {
          totalVoltage += currentState.voltage_potential[i][j];
          count++;
        }
      }

      const avgVoltage = totalVoltage / count;

      // Calculate a simple "complexity" metric - standard deviation of voltage
      let sumSquaredDiff = 0;
      for (let i = 0; i < currentState.voltage_potential.length; i++) {
        for (let j = 0; j < currentState.voltage_potential[i].length; j++) {
          sumSquaredDiff += Math.pow(
            currentState.voltage_potential[i][j] - avgVoltage,
            2
          );
        }
      }

      const complexity = Math.sqrt(sumSquaredDiff / count);

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

    console.log("Visualization updated successfully");
  } catch (error) {
    console.error("Error updating visualization:", error);
  }
}

// Run a single step when the page loads to initialize
$(document).ready(function () {
  // Wait a bit for everything to load
  setTimeout(function () {
    runStep();
  }, 1000);
});
