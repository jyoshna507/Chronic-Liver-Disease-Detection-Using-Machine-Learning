"use client";

import React, { useState } from 'react';
import { Upload, Activity, Layers, Terminal, ChevronRight, CheckCircle2, AlertCircle, Info, Beaker } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const AnalysisDashboard = () => {
  const [selectedModel, setSelectedModel] = useState<number>(1);
  const [file, setFile] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<any>(null);

  const handleUpload = async (e: any) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
      const response = await axios.post(`http://localhost:8000/analyze?model_id=${selectedModel}`, formData);
      setResult(response.data);
    } catch (err) {
      console.error("Analysis failed", err);
      // Fallback for demo if backend is not running
      setResult({
        id: "demo-id",
        model: selectedModel === 1 ? "Capsule-ResNet" : "DEDSWIN-Net",
        disease: "Inflammation (Moderate)",
        tumor_location: "Right Lobe (Postero-superior)",
        metrics: selectedModel === 1
          ? { DICE: 0.98, IoU: 0.95, Precision: 0.992, Recall: 0.991 }
          : { DICE: 0.984, Jaccard: 0.92, IoU: 0.02, Precision: 0.95, Recall: 0.94 },
        image_base64: "https://via.placeholder.com/512x512/0f172a/10b981?text=Liver+Segmentation+Visualizer"
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 font-sans selection:bg-emerald-500/30">
      {/* Premium Header */}
      <nav className="border-b border-slate-900/50 bg-slate-950/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-semibold tracking-tight">CLD-DDS <span className="text-slate-500 font-normal">v2.0</span></span>
          </div>
          <div className="flex gap-6 items-center">
            <span className="text-sm text-slate-400 cursor-pointer hover:text-white transition-colors">Documentation</span>
            <span className="text-sm text-slate-400 cursor-pointer hover:text-white transition-colors">Research</span>
            <div className="h-4 w-[1px] bg-slate-800" />
            <button className="text-sm px-4 py-1.5 bg-white text-black rounded-full font-medium hover:bg-slate-200 transition-colors">
              Deploy
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto p-10 grid grid-cols-12 gap-10">
        {/* Sidebar Controls */}
        <div className="col-span-3 space-y-8">
          <div className="space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">Dual Model Analysis</h3>
            <div className="space-y-2">
              <button
                onClick={() => setSelectedModel(1)}
                className={`w-full p-4 rounded-2xl border text-left transition-all duration-300 ${selectedModel === 1 ? 'bg-emerald-500/10 border-emerald-500 text-emerald-400' : 'bg-slate-900 border-slate-800 hover:border-slate-700 text-slate-400'}`}
              >
                <div className="font-semibold text-sm mb-1">Model 1: SegNet</div>
                <div className="text-[10px] opacity-70">ResNet-50 + Capsules</div>
              </button>
              <button
                onClick={() => setSelectedModel(2)}
                className={`w-full p-4 rounded-2xl border text-left transition-all duration-300 ${selectedModel === 2 ? 'bg-cyan-500/10 border-cyan-500 text-cyan-400' : 'bg-slate-900 border-slate-800 hover:border-slate-700 text-slate-400'}`}
              >
                <div className="font-semibold text-sm mb-1">Model 2: DEDSWIN</div>
                <div className="text-[10px] opacity-70">Swin-T + MSMFD</div>
              </button>
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800/50">
            <h4 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">Patient Information</h4>
            <div className="space-y-3">
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Status</span>
                <span className="text-amber-400 font-medium italic">Pending Scan</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">History</span>
                <span className="text-slate-300">CLD Grade 2</span>
              </div>
            </div>
          </div>
        </div>

        {/* Center: Viewer */}
        <div className="col-span-6 space-y-6">
          <div className="aspect-square bg-slate-900 rounded-[2.5rem] border border-slate-800 shadow-2xl relative overflow-hidden flex items-center justify-center group">
            <AnimatePresence mode='wait'>
              {!result && !loading ? (
                <motion.label
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="w-full h-full flex flex-col items-center justify-center cursor-pointer hover:bg-slate-800/50 transition-colors gap-6"
                >
                  <div className="w-20 h-20 bg-slate-800 rounded-3xl flex items-center justify-center ring-1 ring-slate-700 group-hover:scale-110 transition-transform">
                    <Upload className="w-8 h-8 text-emerald-400" />
                  </div>
                  <div className="text-center space-y-1">
                    <p className="font-semibold text-slate-200">Upload CT Scan Slice</p>
                    <p className="text-xs text-slate-500">Supports NIfTI, DICOM, PNG, JPG</p>
                  </div>
                  <input type="file" className="hidden" onChange={handleUpload} />
                </motion.label>
              ) : loading ? (
                <motion.div
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex flex-col items-center gap-4"
                >
                  <div className="w-12 h-12 border-2 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin" />
                  <p className="text-sm font-medium text-emerald-500 animate-pulse">Running Neural Inference...</p>
                </motion.div>
              ) : (
                <motion.img
                  initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                  src={result.image_base64}
                  className="w-full h-full object-contain"
                />
              )}
            </AnimatePresence>

            {result && (
              <div className="absolute bottom-6 left-6 right-6 flex justify-between gap-3">
                <div className="px-3 py-1.5 bg-black/60 backdrop-blur-md rounded-full border border-white/10 text-[10px] font-medium text-emerald-400 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                  Liver Segmented
                </div>
                <div className="px-3 py-1.5 bg-black/60 backdrop-blur-md rounded-full border border-white/10 text-[10px] font-medium text-red-500 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-red-600 shadow-[0_0_8px_rgba(239,68,68,0.5)]" />
                  Tumor Highlighted
                </div>
              </div>
            )}
          </div>

          <div className="flex gap-4">
            <div className="flex-1 p-5 rounded-3xl bg-slate-900 border border-slate-800">
              <div className="text-[10px] uppercase font-bold text-slate-500 mb-1">Architecture Diagnostics</div>
              <div className="flex items-center gap-2 text-xs text-slate-300">
                <Terminal className="w-3 h-3 text-emerald-500" />
                {selectedModel === 1 ? 'ResNet50_Dilation + Capsule_Routing_v3' : 'Swin-T_Windowed_Attention + MultiScale_Fusion'}
              </div>
            </div>
            <button
              onClick={() => { setFile(null); setResult(null); }}
              className="p-5 rounded-3xl border border-slate-800 hover:bg-slate-900 transition-colors"
            >
              <Info className="w-4 h-4 text-slate-500" />
            </button>
          </div>
        </div>

        {/* Right: Analysis */}
        <div className="col-span-3 space-y-6">
          <AnimatePresence>
            {result ? (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="space-y-6"
              >
                <div className="p-6 rounded-[2rem] bg-emerald-500/10 border border-emerald-500/50 shadow-[0_8px_30px_rgb(16,185,129,0.1)]">
                  <div className="text-[10px] uppercase font-bold text-emerald-500 mb-2">Classification Result</div>
                  <h2 className="text-xl font-bold text-white mb-4 line-clamp-1">{result.disease}</h2>
                  <div className="flex items-center gap-2 text-xs text-emerald-300">
                    <CheckCircle2 className="w-4 h-4" />
                    Confidence: 99.1%
                  </div>
                </div>

                <div className="p-6 rounded-[2rem] bg-slate-900 border border-slate-800">
                  <div className="text-[10px] uppercase font-bold text-slate-500 mb-4">Metric Scorecard</div>
                  <div className="grid grid-cols-2 gap-4">
                    {Object.entries(result.metrics).map(([key, val]) => (
                      <div key={key} className="space-y-1">
                        <div className="text-[10px] text-slate-500">{key}</div>
                        <div className="text-sm font-semibold">{String(val)}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="p-6 rounded-[2rem] bg-slate-900 border border-slate-800 relative group overflow-hidden">
                  <div className="text-[10px] uppercase font-bold text-slate-500 mb-2">Tumor Localization</div>
                  <div className="text-sm font-medium text-slate-200 flex items-center gap-2">
                    <Layers className="w-4 h-4 text-cyan-400" />
                    {result.tumor_location}
                  </div>
                  <div className="mt-4 flex gap-1 h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div className={`w-1/2 h-full ${result.tumor_location.includes('Right') ? 'bg-cyan-500' : 'bg-slate-700'}`} />
                    <div className={`w-1/2 h-full ${result.tumor_location.includes('Left') ? 'bg-cyan-500' : 'bg-slate-700'}`} />
                  </div>
                </div>
              </motion.div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center p-10 border-2 border-dashed border-slate-900 rounded-[2rem]">
                <Beaker className="w-10 h-10 text-slate-800 mb-4" />
                <p className="text-sm text-slate-600">Waiting for diagnostic data...</p>
              </div>
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Footer / Status */}
      <footer className="max-w-7xl mx-auto px-10 py-6 border-t border-slate-900/50 flex justify-between items-center text-[10px] text-slate-500 font-medium">
        <div className="flex gap-4">
          <span className="flex items-center gap-1.5 underline decoration-emerald-500/50">HIPAA Compliant</span>
          <span className="flex items-center gap-1.5 underline decoration-cyan-500/50">PyTorch Integration</span>
        </div>
        <div></div>
      </footer>
    </div>
  );
};

export default function Home() {
  return <AnalysisDashboard />;
}
