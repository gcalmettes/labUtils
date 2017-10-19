//tolerance for line simplification
var toleranceValue = 0.02;

var svg = d3.select("svg"),
    margin = {top: 20, right: 20, bottom: 110, left: 50},
    margin2 = {top: 430, right: 20, bottom: 30, left: 40},
    width = +svg.attr("width") - margin.left - margin.right,
    height = +svg.attr("height") - margin.top - margin.bottom,
    height2 = +svg.attr("height") - margin2.top - margin2.bottom;

var x = d3.scaleLinear().range([0, width]),
    x2 = d3.scaleLinear().range([0, width]),
    y = d3.scaleLinear().range([height, 0]),
    y2 = d3.scaleLinear().range([height2, 0]);

var xAxis = d3.axisBottom(x),
    xAxis2 = d3.axisBottom(x2),
    yAxis = d3.axisLeft(y);

var brush = d3.brushX()
    .extent([[0, 0], [width, height2]])
    .on("brush end", brushed);

var zoom = d3.zoom()
    .scaleExtent([1, Infinity])
    .translateExtent([[0, 0], [width, height]])
    .extent([[0, 0], [width, height]])
    .on("zoom", zoomed);

var line = d3.line()
    .curve(d3.curveMonotoneX)
    .x(function(d) { return x(d.x); })
    .y(function(d) { return y(d.y); });

var line2 = d3.line()
    .curve(d3.curveMonotoneX)
    .x(function(d) { return x2(d.x); })
    .y(function(d) { return y2(d.y); });

svg.append("defs").append("clipPath")
    .attr("id", "clip")
  .append("rect")
    .attr("width", width)
    .attr("height", height);

var focus = svg.append("g")
    .attr("class", "focus")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var context = svg.append("g")
    .attr("class", "context")
    .attr("transform", "translate(" + margin2.left + "," + margin2.top + ")");

var rawData = []
var data = []

//import data, apply type conversion and draw initial graph
d3.csv("data/APTrace.csv", type, drawGraph)

function drawGraph(error, inputData) {
  if (error) throw error;

  //line simplification
  rawData = inputData
  data = simplifyData(rawData, toleranceValue)
  
  var deltaData = d3.max(data, function(d) { return d.y; }) - d3.min(data, function(d) { return d.y; })
  var cushion = 0.08 * deltaData
  //scales based on data
  //focus scale (for yScale: min/max of data +/- 8% of delta)
  x.domain(d3.extent(data, function(d) { return d.x; }));
  y.domain([d3.min(data, function(d) { return d.y; })-cushion, d3.max(data, function(d) { return d.y; })+cushion]);
  //context scale
  x2.domain(x.domain());
  y2.domain(y.domain());
  

  focus.append("path")
      .datum(data)
      .attr("class", "dataLine")
      .attr("d", line)
      .attr("clip-path", "url(#clip)");

  focus.append("g")
      .attr("class", "axis axis--x")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis);

  focus.append("g")
      .attr("class", "axis axis--y")
      .call(yAxis);
  focus.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Voltage")
      .style("font-size", 18);

  context.append("path")
      .datum(data)
      .attr("class", "dataLine")
      .attr("d", line2);

  context.append("g")
      .attr("class", "axis axis--x")
      .attr("transform", "translate(0," + height2 + ")")
      .call(xAxis2);

  context.append("g")
      .attr("class", "xbrush")
      .call(brush);
      // .call(brush.move, x.range());

  svg.append("rect")
      .attr("class", "zoom")
      .attr("width", width)
      .attr("height", height)
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
      .call(zoom);

};

// tolerance value from slider
d3.select("#toleranceSlider")
  .on("input", updateData)

d3.select("#myFile")
  .on("change", changeData)

function changeData(){
  var file = d3.event.target.files[0];
  if (file) {
    var reader = new FileReader();
      reader.onloadend = function(evt) {
        var dataUrl = evt.target.result;
        // The following call results in an "Access denied" error in IE.
        loadCSV(dataUrl);
    };
   reader.readAsDataURL(file);
  }
}

function loadCSV(csvUrl) {
   d3.csv(csvUrl, type, loadNewData);
 }

 function loadNewData(newData){
  rawData = newData;
  data = simplifyData(rawData, toleranceValue);

  var deltaData = d3.max(data, function(d) { return d.y; }) - d3.min(data, function(d) { return d.y; })
  var cushion = 0.08 * deltaData
  //scales based on data
  //focus scale (for yScale: min/max of data +/- 8% of delta)
  x.domain(d3.extent(data, function(d) { return d.x; }));
  y.domain([d3.min(data, function(d) { return d.y; })-cushion, d3.max(data, function(d) { return d.y; })+cushion]);
  //context scale
  x2.domain(x.domain());
  y2.domain(y.domain());

  //reinitialize brush on context
  context.select(".xbrush").remove();
  context.append("g")
      .attr("class", "xbrush")
      .call(brush);

  focus.select("path")
      .datum(data)
      .transition()
        .attr("d", line);

  context.select("path")
      .datum(data)
      .transition()
        .attr("d", line2);

  context.select(".axis--x").call(xAxis2);
  focus.select(".axis--x").call(xAxis);
  focus.select(".axis--y").call(yAxis);

  updateData();
 }

var formatToleranceValue = d3.format(",.3f")

//// If performance needed, can implement throttle for line simplification rendering
// var free = true;
// function updateData() {
//     var throttleWaitingTime = 300
//     if (free){
//       value = d3.select("#toleranceSlider").node().value
//       toleranceValue = +value;
//       d3.select("#toleranceLabel").text( d => formatToleranceValue(+value));
//       data = simplifyData(rawData, toleranceValue)
//       render(data)
//       free = false;
//       setTimeout(function(){
//         free = true;
//       }, throttleWaitingTime) 
//     } else {
//       return
//     }
// };

function updateData() {
  value = d3.select("#toleranceSlider").node().value
  toleranceValue = +value;
  d3.select("#toleranceLabel").text( d => formatToleranceValue(+value));
  data = simplifyData(rawData, toleranceValue)
  render(data)
};

function render(data) {
  d3.select(".dataLine")
    .datum(data)
    .transition()
      .attrTween("d", lineTween);
}

function brushed(){
  if (d3.event.sourceEvent && d3.event.sourceEvent.type === "zoom") return; // ignore brush-by-zoom
  brushSpan = d3.brushSelection(this)
  x.domain(brushSpan === null ? x2.domain() : [x2.invert(brushSpan[0]), x2.invert(brushSpan[1])]);
  focus.selectAll("path.dataLine").attr("d",  function(d) {
    return line(d)
  });
  focus.select(".axis.axis--x").call(xAxis);
  focus.select(".axis.axis--y").call(yAxis);
  svg.select(".zoom").call(zoom.transform, d3.zoomIdentity
      .scale(width / (brushSpan[1] - brushSpan[0]))
      .translate(-brushSpan[0], 0));
}

function zoomed() {
  if (d3.event.sourceEvent && d3.event.sourceEvent.type === "brush") return; // ignore zoom-by-brush
  var t = d3.event.transform;
  x.domain(t.rescaleX(x2).domain());
  focus.select(".dataLine").attr("d", line);
  focus.select(".axis--x").call(xAxis);
  context.select(".xbrush").call(brush.move, x.range().map(t.invertX, t));
}

function type(d) {
  d.x = +d.x;
  d.y = +d.y;
  return d;
}

function lineTween(d) {
  return t => {
    return line(d);
  }
}


function simplifyData(data, tolerance) {
  //line simplification via simplify.js
  data = simplify(data, tolerance, true);
  return data
}



