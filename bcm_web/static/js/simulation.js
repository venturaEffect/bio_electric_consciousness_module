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
});

function initPlots() {
  console.log("Initializing plots...");

  // Initialize main visualization
  Plotly.newPlot(
    "visualization",
    [
      {
        z: Array(10)
          .fill()
          .map(() => Array(10).fill(0)),
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
    }
  );

  console.log("Plots initialized");
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
    $.post(
      "/api/modify_cell",
      {
        x: y, // Swap x/y for correct orientation in heatmap
        y: x,
        ion: currentView === "voltage" ? "voltage" : currentView,
        intensity: intensity,
      },
      function (response) {
        if (response.success) {
          currentState = response.state;
          updateVisualization();
        }
      }
    );
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
  $.post(
    "/api/init_simulation",
    {
      scenario: scenario,
    },
    function (response) {
      if (response.success) {
        currentState = response.state;
        updateVisualization();
      }
    }
  );
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
  $.post(
    "/api/run_step",
    {
      config_updates: configUpdates,
      state: currentState,
      scenario: scenario,
    },
    function (response) {
      if (response.success) {
        // Update state
        currentState = response.state;

        // Update iteration count
        iterationCount++;
        $("#iteration-counter").text("Iteration: " + iterationCount);

        // Update visualization
        updateVisualization();
      }
    }
  );
}

function updateVisualization() {
  console.log("Updating visualization with current state");

  if (!currentState) {
    console.error("No current state available");
    return;
  }

  // Select data based on current view
  let plotData = [];

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
  } else if (currentView === "morphology" && currentState.morphological_state) {
    // Convert 1D morphological state to 2D for visualization
    const size = Math.sqrt(currentState.morphological_state.length);
    let morphGrid = [];

    for (let i = 0; i < size; i++) {
      let row = [];
      for (let j = 0; j < size; j++) {
        row.push(currentState.morphological_state[i * size + j]);
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

  // Update the main visualization
  Plotly.react("visualization", plotData, {
    paper_bgcolor: "#222",
    plot_bgcolor: "#222",
    font: { color: "#fff" },
    margin: { t: 10, l: 50, r: 50, b: 10 },
  });

  // Update time series plot
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

    // Calculate pattern complexity (simplified as standard deviation)
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

    // Update complexity metrics
    Plotly.extendTraces(
      "pattern-metrics",
      {
        x: [[iterationCount]],
        y: [[complexity]],
      },
      [0]
    );

    // Show only the last 50 data points for clarity
    const maxPoints = 50;
    if (iterationCount > maxPoints) {
      Plotly.relayout("time-series", {
        xaxis: {
          range: [iterationCount - maxPoints, iterationCount],
          title: "Iteration",
          gridcolor: "#444",
          color: "#fff",
        },
      });

      Plotly.relayout("pattern-metrics", {
        xaxis: {
          range: [iterationCount - maxPoints, iterationCount],
          title: "Iteration",
          gridcolor: "#444",
          color: "#fff",
        },
      });
    }
  }

  console.log("Visualization updated");
}

// Run a single step when the page loads to initialize
$(document).ready(function () {
  // Wait a bit for everything to load
  setTimeout(function () {
    runStep();
  }, 1000);
});
