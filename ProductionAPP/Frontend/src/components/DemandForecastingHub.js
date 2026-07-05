import React, { useState } from 'react';
import { 
  TrendingUp, 
  HelpCircle, 
  Layers, 
  AlertTriangle, 
  Calendar, 
  Activity, 
  Check, 
  Sliders 
} from 'lucide-react';
import { FESTIVALS } from './mockData';

export default function DemandForecastingHub({ storeInfo }) {
  const [showRawML, setShowRawML] = useState(true);
  const [showAdjusted, setShowAdjusted] = useState(true);
  
  // Custom tooltips
  const [hoveredIndex, setHoveredIndex] = useState(null);

  // Default store details fallback
  const store = storeInfo || {
    name: 'Executive Dashboard',
    type: 'Supermarket',
    location: 'Urban',
    openingMonth: 'October',
    investment: 850000
  };

  // 7-day demand data (rolling)
  // Let's generate a dataset that exhibits capacity clipping
  // If store is Medium format, capacity cap is 800.
  // If store is Small format, capacity cap is 400.
  // We'll show a surge on Day 5 (Friday) and Day 6 (Saturday).
  const getChartData = () => {
    const rawData = [450, 480, 520, 590, 840, 910, 680]; // Raw ML Signal
    const actualData = [430, 470, 510, 570, 780, 800, 650]; // Actual Units Sold
    
    // Apply constraints
    let capacityLimit = 9999;
    if (store.type === 'Small') capacityLimit = 400;
    else if (store.type === 'Medium') capacityLimit = 800;

    const locationMultiplier = store.location === 'Rural' ? 0.6 : store.location === 'Semi-Urban' ? 0.8 : 1.0;

    const adjustedData = rawData.map(val => {
      // Step 1: Apply Location Multiplier
      let demand = val * locationMultiplier;
      // Step 2: Apply Capacity Cap (Decision Intelligence)
      if (demand > capacityLimit) {
        return capacityLimit; // Clipped
      }
      return Math.round(demand);
    });

    return {
      days: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
      raw: rawData,
      actual: actualData,
      adjusted: adjustedData,
      capacity: capacityLimit
    };
  };

  const chart = getChartData();

  // Find peak demand and check if it breaches capacity
  const maxAdjusted = Math.max(...chart.adjusted);
  const isBreached = maxAdjusted >= chart.capacity;
  const breachPercent = Math.min(100, Math.round((maxAdjusted / chart.capacity) * 100));

  // Determine current active festival based on opening month
  const activeFestival = FESTIVALS.find(f => f.month.toLowerCase() === store.openingMonth.toLowerCase());

  // Coordinates mapping for SVG graph (View Box 0 0 500 200)
  // X points from 40 to 460
  // Y points from 160 (value = 0) to 20 (value = 1000)
  const getCoordinates = (data) => {
    const minVal = 0;
    const maxVal = 1000;
    return data.map((val, idx) => {
      const x = 40 + (idx * 70);
      const y = 160 - ((val - minVal) / (maxVal - minVal)) * 140;
      return { x, y, value: val };
    });
  };

  const rawCoords = getCoordinates(chart.raw);
  const actualCoords = getCoordinates(chart.actual);
  const adjustedCoords = getCoordinates(chart.adjusted);
  const capY = 160 - ((chart.capacity - 0) / (1000 - 0)) * 140;

  const buildPath = (coords) => {
    return coords.reduce((acc, coord, idx) => {
      return idx === 0 ? `M ${coord.x} ${coord.y}` : `${acc} L ${coord.x} ${coord.y}`;
    }, '');
  };

  return (
    <div className="space-y-6 font-sans">
      {/* Top Header Row */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-emerald-500" /> Demand Forecasting Hub
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Real-time daily item demand forecasts layered with system intelligence constraints.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-zinc-500 uppercase font-semibold text-[10px] tracking-wider">Active Store:</span>
          <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-bold uppercase tracking-wider">
            {store.type} - {store.location} | Active Node
          </span>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left 2 Columns: Charting Suite */}
        <div className="lg:col-span-2 bg-zinc-900/40 border border-zinc-800 rounded-2xl p-5 flex flex-col gap-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-zinc-800">
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">Demand Charting Suite</h3>
              <p className="text-[10px] text-zinc-500">7-Day rolling timeline. Comparison of raw signals vs production adjustments.</p>
            </div>
            
            {/* Toggles */}
            <div className="flex items-center gap-2 text-xs">
              <button
                onClick={() => setShowRawML(!showRawML)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border transition-all ${
                  showRawML 
                    ? 'bg-blue-500/10 border-blue-500/30 text-blue-400 font-bold' 
                    : 'bg-zinc-950 border-zinc-800 text-zinc-500'
                }`}
              >
                <div className={`w-1.5 h-1.5 rounded-full ${showRawML ? 'bg-blue-400 animate-pulse' : 'bg-zinc-600'}`} />
                <span>Raw ML Signal</span>
              </button>

              <button
                onClick={() => setShowAdjusted(!showAdjusted)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border transition-all ${
                  showAdjusted 
                    ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 font-bold' 
                    : 'bg-zinc-950 border-zinc-800 text-zinc-500'
                }`}
              >
                <div className={`w-1.5 h-1.5 rounded-full ${showAdjusted ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-600'}`} />
                <span>Adjusted Output</span>
              </button>
            </div>
          </div>

          {/* SVG Chart */}
          <div className="relative bg-zinc-950/60 border border-zinc-800/80 rounded-xl p-4 flex-1 min-h-[220px]">
            <svg viewBox="0 0 500 200" className="w-full h-full">
              {/* Horizontal gridlines */}
              <line x1="40" y1="160" x2="460" y2="160" stroke="#1f1f23" strokeWidth="1" />
              <line x1="40" y1="125" x2="460" y2="125" stroke="#1f1f23" strokeWidth="0.8" strokeDasharray="2 2" />
              <line x1="40" y1="90" x2="460" y2="90" stroke="#1f1f23" strokeWidth="0.8" strokeDasharray="2 2" />
              <line x1="40" y1="55" x2="460" y2="55" stroke="#1f1f23" strokeWidth="0.8" strokeDasharray="2 2" />
              <line x1="40" y1="20" x2="460" y2="20" stroke="#1f1f23" strokeWidth="1" />

              {/* Chart labels for Y Axis */}
              <text x="32" y="163" fill="#52525b" fontSize="7" textAnchor="end">0</text>
              <text x="32" y="128" fill="#52525b" fontSize="7" textAnchor="end">250</text>
              <text x="32" y="93" fill="#52525b" fontSize="7" textAnchor="end">500</text>
              <text x="32" y="58" fill="#52525b" fontSize="7" textAnchor="end">750</text>
              <text x="32" y="23" fill="#52525b" fontSize="7" textAnchor="end">1000</text>

              {/* Days labels */}
              {chart.days.map((day, idx) => (
                <text key={day} x={40 + (idx * 70)} y="176" fill="#71717a" fontSize="8" textAnchor="middle" fontWeight="semibold">
                  {day}
                </text>
              ))}

              {/* Physical Storage Cap Line (Flashed red/amber depending on breach) */}
              {chart.capacity < 1000 && (
                <g>
                  <line 
                    x1="40" 
                    y1={capY} 
                    x2="460" 
                    y2={capY} 
                    stroke={isBreached ? '#f43f5e' : '#f59e0b'} 
                    strokeWidth="1" 
                    strokeDasharray="4 2" 
                  />
                  <text 
                    x="455" 
                    y={capY - 4} 
                    fill={isBreached ? '#f43f5e' : '#f59e0b'} 
                    fontSize="6" 
                    fontWeight="bold" 
                    textAnchor="end"
                  >
                    CAPACITY CAP: {chart.capacity} UNITS
                  </text>
                </g>
              )}

              {/* Lines & Nodes */}
              {/* Actual Sales Line (always showing as thin baseline) */}
              <path d={buildPath(actualCoords)} fill="none" stroke="#71717a" strokeWidth="1.2" strokeDasharray="3 3" opacity="0.6" />
              {actualCoords.map((coord, idx) => (
                <circle 
                  key={`act-${idx}`} 
                  cx={coord.x} 
                  cy={coord.y} 
                  r="2" 
                  fill="#71717a" 
                  onMouseEnter={() => setHoveredIndex({ type: 'Actual', val: coord.value, day: chart.days[idx], x: coord.x, y: coord.y })}
                  onMouseLeave={() => setHoveredIndex(null)}
                  className="cursor-pointer"
                />
              ))}

              {/* Raw ML Line */}
              {showRawML && (
                <>
                  <path d={buildPath(rawCoords)} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" />
                  {rawCoords.map((coord, idx) => (
                    <circle
                      key={`raw-${idx}`}
                      cx={coord.x}
                      cy={coord.y}
                      r="3.5"
                      fill="#1e3a8a"
                      stroke="#3b82f6"
                      strokeWidth="1.5"
                      onMouseEnter={() => setHoveredIndex({ type: 'Raw ML Signal', val: coord.value, day: chart.days[idx], x: coord.x, y: coord.y })}
                      onMouseLeave={() => setHoveredIndex(null)}
                      className="cursor-pointer hover:scale-125 transition-transform"
                    />
                  ))}
                </>
              )}

              {/* Adjusted Production Output Line */}
              {showAdjusted && (
                <>
                  <path d={buildPath(adjustedCoords)} fill="none" stroke="#10b981" strokeWidth="2.5" strokeLinecap="round" />
                  {adjustedCoords.map((coord, idx) => {
                    const isClipped = coord.value >= chart.capacity;
                    return (
                      <circle
                        key={`adj-${idx}`}
                        cx={coord.x}
                        cy={coord.y}
                        r="4"
                        fill={isClipped ? '#9f1239' : '#064e3b'}
                        stroke={isClipped ? '#f43f5e' : '#10b981'}
                        strokeWidth="1.8"
                        onMouseEnter={() => setHoveredIndex({ 
                          type: 'Adjusted Production Output', 
                          val: coord.value, 
                          day: chart.days[idx], 
                          x: coord.x, 
                          y: coord.y,
                          clipped: isClipped,
                          rawVal: chart.raw[idx]
                        })}
                        onMouseLeave={() => setHoveredIndex(null)}
                        className="cursor-pointer hover:scale-125 transition-transform"
                      />
                    );
                  })}
                </>
              )}
            </svg>

            {/* Custom Interactive Tooltip HTML */}
            {hoveredIndex && (
              <div 
                className="absolute z-10 bg-zinc-950 border border-zinc-800 p-2.5 rounded-lg text-[10px] space-y-1 shadow-xl pointer-events-none"
                style={{ 
                  left: `${(hoveredIndex.x / 500) * 100}%`, 
                  top: `${(hoveredIndex.y / 200) * 100 - 30}%`,
                  transform: 'translate(-50%, -100%)' 
                }}
              >
                <div className="flex items-center gap-1.5 justify-between">
                  <span className="font-bold text-zinc-100">{hoveredIndex.day}</span>
                  <span className="text-[8px] px-1 rounded bg-zinc-900 text-zinc-500 font-bold uppercase">{hoveredIndex.type}</span>
                </div>
                <div>
                  <span className="text-zinc-400">Demand:</span>{' '}
                  <span className="font-semibold text-white">{hoveredIndex.val} units</span>
                </div>
                {hoveredIndex.clipped && (
                  <div className="text-[9px] text-rose-400 font-semibold border-t border-zinc-850 pt-1 mt-1 flex items-center gap-1">
                    <AlertTriangle className="w-3 h-3 text-rose-500 shrink-0" />
                    <span>Clipped by physical format limit (Raw: {hoveredIndex.rawVal})</span>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="flex items-center justify-between text-[10px] text-zinc-500 border-t border-zinc-850 pt-3">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-blue-500" />
                <span>Raw prediction without local scaling</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-emerald-500" />
                <span>Adjusted output with constraints</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-0.5 border-t border-dashed border-zinc-500" />
                <span>Historical actual sales</span>
              </div>
            </div>
            <span>*Based on base Hybrid Ensemble weighting (0.923 R²)</span>
          </div>

        </div>

        {/* Right 1 Column: Metric Gauges & Seasonality */}
        <div className="space-y-6">
          
          {/* Gauge: Storage volume limit indicator */}
          <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-5 space-y-4">
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">Storage Volume Gauge</h3>
              <p className="text-[10px] text-zinc-500">Physical constraint checks from the Decision Intelligence layer.</p>
            </div>

            <div className="flex items-center gap-4">
              {/* Circular Progress Gauge */}
              <div className="relative w-16 h-16 flex items-center justify-center shrink-0">
                <svg width="64" height="64" viewBox="0 0 36 36" className="transform -rotate-90">
                  <circle cx="18" cy="18" r="16" fill="none" stroke="#27272a" strokeWidth="3" />
                  <circle 
                    cx="18" 
                    cy="18" 
                    r="16" 
                    fill="none" 
                    stroke={isBreached ? '#f43f5e' : '#10b981'} 
                    strokeWidth="3.2" 
                    strokeDasharray={`${breachPercent} 100`} 
                    strokeLinecap="round"
                    className="transition-all duration-500"
                  />
                </svg>
                <span className={`absolute text-xs font-bold ${isBreached ? 'text-rose-400' : 'text-emerald-400'}`}>
                  {breachPercent}%
                </span>
              </div>

              <div className="space-y-1.5">
                <div className="text-xs">
                  <span className="text-zinc-400">Peak Demand:</span>{' '}
                  <span className="font-bold text-white">{maxAdjusted} units</span>
                </div>
                <div className="text-xs">
                  <span className="text-zinc-400">Storage Cap:</span>{' '}
                  <span className="font-bold text-zinc-300">{chart.capacity} items/day</span>
                </div>
              </div>
            </div>

            {isBreached ? (
              <div className="p-3 rounded-lg bg-rose-950/20 border border-rose-900/30 text-[10px] text-rose-300 flex gap-2">
                <AlertTriangle className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
                <div className="space-y-0.5">
                  <span className="font-bold">Capacity Breach Alert!</span>
                  <p className="text-zinc-400">Decision Intelligence layer has capped incoming order pipelines to avoid physical inventory cluttering.</p>
                </div>
              </div>
            ) : (
              <div className="p-3 rounded-lg bg-emerald-950/15 border border-emerald-900/30 text-[10px] text-emerald-300 flex gap-2">
                <Check className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                <div className="space-y-0.5">
                  <span className="font-bold">Intake Capacity Cleared</span>
                  <p className="text-zinc-400">Forecasted sales stay within the physical constraints of a {store.type} format.</p>
                </div>
              </div>
            )}
          </div>

          {/* Card: Seasonal festival impact tracker */}
          <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white uppercase tracking-wider">Seasonal Impact Tracker</h3>
                <p className="text-[10px] text-zinc-500">Active regional calendar scaling multipliers.</p>
              </div>
              <Calendar className="w-4 h-4 text-zinc-500" />
            </div>

            {activeFestival ? (
              <div className="p-4 rounded-xl bg-zinc-950 border border-emerald-500/25 space-y-3 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/5 rounded-full blur-2xl pointer-events-none" />
                
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-white">{activeFestival.name} Festival Overlay</span>
                  <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-bold">
                    +{Math.round((activeFestival.modifier - 1) * 100)}% Surge Active
                  </span>
                </div>
                
                <p className="text-[11px] text-zinc-400 leading-relaxed">
                  {activeFestival.description}. Demand models automatically upscale staples and perishables to absorb peak festival velocity.
                </p>
                
                <div className="text-[9px] text-zinc-500 font-semibold uppercase tracking-wider flex items-center gap-1.5 pt-1 border-t border-zinc-900">
                  <Activity className="w-3.5 h-3.5 text-emerald-500" /> 
                  <span>Active scaling modifier: {activeFestival.modifier}x</span>
                </div>
              </div>
            ) : (
              <div className="p-4 rounded-xl bg-zinc-950 border border-zinc-850 text-xs text-zinc-400 text-center py-6">
                <Calendar className="w-8 h-8 text-zinc-700 mx-auto mb-2" />
                <span>No major cultural festivals active in {store.openingMonth} calendar.</span>
                <span className="block text-[9px] text-zinc-600 mt-1">Standard seasonal baseline indices applied.</span>
              </div>
            )}
            
            <div className="space-y-2">
              <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider block">Upcoming Core Calendar Events</span>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {FESTIVALS.map(f => (
                  <div key={f.name} className="p-2 rounded bg-zinc-950 border border-zinc-900 flex justify-between items-center">
                    <div>
                      <span className="font-semibold text-zinc-300 block text-[10px]">{f.name}</span>
                      <span className="text-[9px] text-zinc-500">{f.month}</span>
                    </div>
                    <span className="text-[10px] font-bold text-emerald-400">+{Math.round((f.modifier - 1) * 100)}%</span>
                  </div>
                ))}
              </div>
            </div>

          </div>

        </div>

      </div>
    </div>
  );
}
