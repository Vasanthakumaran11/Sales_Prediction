import React, { useState } from 'react';
import { 
  Cpu, 
  Database, 
  RotateCw, 
  TrendingUp, 
  CheckCircle, 
  Play, 
  Layers, 
  Info, 
  Flame 
} from 'lucide-react';
import { ALGORITHMS } from './mockData';

export default function ModelLearningLoop() {
  const [retrainingDays, setRetrainingDays] = useState(11);
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainStep, setRetrainStep] = useState('');
  const [isCompleted, setIsCompleted] = useState(false);
  const [customR2, setCustomR2] = useState('0.9228 (Base Model)');

  const handleRetrain = () => {
    setIsRetraining(true);
    setIsCompleted(false);
    
    // Animate the training steps
    const steps = [
      { msg: 'Ingesting daily transaction caches...', delay: 800 },
      { msg: 'Engineering lag features (1-day, 7-day)...', delay: 1800 },
      { msg: 'Tuning hyperparameters for 7 regression layers...', delay: 3000 },
      { msg: 'Recalibrating stacking weights (Hybrid Ensemble)...', delay: 4200 },
      { msg: 'Serializing personalized_model.pkl binaries...', delay: 5200 }
    ];

    steps.forEach(step => {
      setTimeout(() => {
        setRetrainStep(step.msg);
      }, step.delay);
    });

    // Complete Training Loop
    setTimeout(() => {
      setIsRetraining(false);
      setIsCompleted(true);
      setCustomR2('0.9412 (Personalized Model)');
      setRetrainingDays(14); // fully filled
    }, 6000);
  };

  return (
    <div className="space-y-6 font-sans">
      {/* Top Header Row */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <Cpu className="w-5 h-5 text-emerald-500" /> Algorithmic Metrics & Learning
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Monitor ML algorithms training logs, check evaluation leaderboards, and trigger personalized store training loops.
          </p>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Retraining progression buffer */}
        <div className="lg:col-span-1 space-y-6">
          
          {/* Data Gathering Progression Buffer Card */}
          <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-5 space-y-5">
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">Progression Buffer</h3>
              <p className="text-[10px] text-zinc-500">Real transaction logging threshold required to run personalization pipelines.</p>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center text-xs">
                <span className="text-zinc-400">Data collection status:</span>
                <span className="font-bold text-white">{retrainingDays} / 14 Days</span>
              </div>

              {/* Progress Slider */}
              <div className="w-full h-3 bg-zinc-950 border border-zinc-850 rounded-full overflow-hidden relative">
                <div 
                  className="h-full rounded-full bg-emerald-500 transition-all duration-700"
                  style={{ width: `${(retrainingDays / 14) * 100}%` }}
                />
              </div>

              <p className="text-[10px] text-zinc-550 leading-relaxed">
                The base model uses general grocery transaction patterns. Retraining personalized layers on your store require a minimum 14-day threshold of active logs.
              </p>
            </div>

            {/* Action Trigger button */}
            {!isRetraining && !isCompleted ? (
              <button
                id="btn-trigger-retrain"
                onClick={handleRetrain}
                className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs rounded-lg transition-all flex items-center justify-center gap-1.5 shadow-lg"
              >
                <Play className="w-4 h-4 text-emerald-300" /> Trigger Personalized Training Loop
              </button>
            ) : isRetraining ? (
              <div className="space-y-3">
                <div className="p-3.5 rounded-lg bg-zinc-950 border border-zinc-850 text-[10px] text-zinc-300 flex items-center gap-2">
                  <RotateCw className="w-4 h-4 text-emerald-400 animate-spin" />
                  <span className="font-medium text-emerald-400 truncate">{retrainStep || 'Initializing pipeline...'}</span>
                </div>
                <div className="w-full bg-zinc-950 h-1.5 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 animate-pulse w-3/4" />
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-emerald-950/20 border border-emerald-500/20 text-xs text-center space-y-2 py-5">
                  <CheckCircle className="w-8 h-8 text-emerald-500 mx-auto" />
                  <div className="space-y-0.5">
                    <span className="font-bold text-white block">Personalization Active!</span>
                    <span className="text-[10px] text-zinc-400">Models optimized to your store demographics.</span>
                  </div>
                </div>
                <button
                  onClick={() => {
                    setIsCompleted(false);
                    setRetrainingDays(11);
                    setCustomR2('0.9228 (Base Model)');
                  }}
                  className="w-full text-center text-[10px] text-zinc-500 hover:text-zinc-400"
                >
                  Reset Retraining Demo
                </button>
              </div>
            )}
          </div>

          {/* Stacking Ensemble Weighting Info */}
          <div className="p-4 rounded-2xl bg-zinc-900/40 border border-zinc-800 space-y-3">
            <div className="flex items-center gap-2 text-white">
              <Layers className="w-4 h-4 text-blue-400" />
              <h4 className="text-xs font-bold uppercase tracking-wider">Stacking Weights Model</h4>
            </div>
            <p className="text-[10px] text-zinc-400 leading-relaxed">
              The Hybrid Ensemble weights individual predictions dynamically based on test metrics. Top tree-based models (Random Forest, XGBoost) inherit higher coefficients during final aggregate calculations.
            </p>
            <div className="text-[9px] text-emerald-400 font-bold uppercase flex items-center gap-1.5 pt-2 border-t border-zinc-850">
              <Flame className="w-3.5 h-3.5 text-orange-500" /> 
              <span>Stacking R² Accuracy: {customR2}</span>
            </div>
          </div>

        </div>

        {/* Right 2 Columns: Algorithmic leaderboard */}
        <div className="lg:col-span-2 bg-zinc-900/40 border border-zinc-800 rounded-2xl p-5 space-y-4">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider font-semibold">Algorithms Leaderboard</h3>
            <p className="text-[10px] text-zinc-500">Evaluation logs compared across standard split datasets (80% Train, 20% Test).</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            
            {/* Hybrid Ensemble Master Card */}
            <div className="bg-zinc-950 border border-emerald-500/25 p-4 rounded-xl relative overflow-hidden flex flex-col justify-between min-h-[140px]">
              <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/5 rounded-full blur-2xl pointer-events-none" />
              <div>
                <span className="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[8px] font-bold uppercase tracking-wider">
                  Master Model
                </span>
                <h4 className="text-sm font-bold text-white mt-2">Hybrid Ensemble Stacker</h4>
                <p className="text-[9px] text-zinc-500 mt-1">Combines all 7 algorithms using test R² coefficients.</p>
              </div>

              <div className="flex justify-between items-end border-t border-zinc-900 pt-3 mt-4">
                <div>
                  <span className="text-[8px] text-zinc-500 uppercase font-semibold">Stability Score</span>
                  <span className="text-sm font-bold text-white block">Optimal Variance</span>
                </div>
                <div className="text-right">
                  <span className="text-[8px] text-zinc-500 uppercase font-semibold">R² Coefficient</span>
                  <span className="text-base font-extrabold text-emerald-400 block">{customR2.split(' ')[0]}</span>
                </div>
              </div>
            </div>

            {/* Individual algorithms leaderboard */}
            <div className="sm:col-span-2 overflow-x-auto border border-zinc-800 rounded-xl bg-zinc-950">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="bg-zinc-900 border-b border-zinc-800 text-zinc-400 font-semibold text-[9px] uppercase tracking-wider">
                    <th className="p-3">Algorithm</th>
                    <th className="p-3">Type</th>
                    <th className="p-3 text-right">R² Score</th>
                    <th className="p-3 text-right">MAE</th>
                    <th className="p-3 text-right">RMSE</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-900">
                  {ALGORITHMS.map(alg => (
                    <tr key={alg.name} className="hover:bg-zinc-900/20 text-zinc-350">
                      <td className="p-3 font-semibold text-white flex items-center gap-1.5">
                        {alg.name}
                        {alg.isBest && (
                          <span className="px-1.5 py-0.2 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[8px] font-bold rounded">
                            Best Base
                          </span>
                        )}
                      </td>
                      <td className="p-3 text-zinc-500 text-[10px]">{alg.type}</td>
                      <td className="p-3 text-right font-medium text-emerald-400">{alg.r2.toFixed(4)}</td>
                      <td className="p-3 text-right text-zinc-450">{alg.mae.toFixed(2)}</td>
                      <td className="p-3 text-right text-zinc-450">{alg.rmse.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

          </div>

          <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-850 text-[10px] text-zinc-500 flex items-start gap-1.5 leading-relaxed">
            <Info className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
            <p>
              Leaderboard values are evaluated using standard k-fold cross-validation. Personalization incorporates local weights to optimize precision for your store format and regional seasonality.
            </p>
          </div>

        </div>

      </div>
    </div>
  );
}
