"use client";

import React, { useState, useEffect } from 'react';
import Gateway from '../components/Gateway';
import Sidebar from '../components/Sidebar';
import DemandForecastingHub from '../components/DemandForecastingHub';
import InventoryReplenishment from '../components/InventoryReplenishment';
import FinancialAllocation from '../components/FinancialAllocation';
import TransactionsLog from '../components/TransactionsLog';
import Settings from '../components/Settings';

export default function Home() {
  const [setupState, setSetupState] = useState('gateway'); // 'gateway', 'active'
  const [storeInfo, setStoreInfo] = useState(null); // null means Executive Mode
  const [isMultiStoreMode, setIsMultiStoreMode] = useState(false);
  const [activeView, setActiveView] = useState('demand'); // 'demand', 'inventory', 'financial', 'transactions', 'settings'
  const [isLoading, setIsLoading] = useState(false);
  const [theme, setTheme] = useState('dark'); // 'dark', 'light'

  // Apply Light/Dark class to document root for Tailwind dark variants toggle
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  // Trigger loading skeleton screen for view transitions
  const triggerViewLoad = (targetView, action) => {
    setIsLoading(true);
    if (action) action();
    setActiveView(targetView);
    setTimeout(() => {
      setIsLoading(false);
    }, 500); // 500ms transition loading state
  };

  // Handling New Store or Sync Launch
  const handleSelectStore = (storeData) => {
    triggerViewLoad('demand', () => {
      setStoreInfo(storeData);
      setIsMultiStoreMode(false);
      setSetupState('active');
    });
  };

  // Handling Multi-Store Executive Entrance
  const handleSelectChain = () => {
    triggerViewLoad('inventory', () => {
      setStoreInfo(null);
      setIsMultiStoreMode(true);
      setSetupState('active');
    });
  };

  const handleBackToGateway = () => {
    setSetupState('gateway');
    setStoreInfo(null);
    setIsMultiStoreMode(false);
  };

  return (
    <main className="min-h-screen bg-zinc-100 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 transition-colors duration-200 selection:bg-emerald-500/30 selection:text-white">
      {setupState === 'gateway' ? (
        <Gateway 
          onSelectStore={handleSelectStore} 
          onSelectChain={handleSelectChain} 
        />
      ) : (
        <div className="flex">
          {/* Persistent Navigation Sidebar */}
          <Sidebar 
            activeView={activeView} 
            setActiveView={(view) => triggerViewLoad(view)} 
            storeInfo={storeInfo}
            onBackToGateway={handleBackToGateway}
          />

          {/* Active Operations View Panel */}
          <div className="flex-1 ml-64 min-h-screen p-8 bg-zinc-100 dark:bg-zinc-950 transition-colors duration-200">
            <div className="max-w-7xl mx-auto space-y-6">
              
              {/* Skeleton Loader for Network States */}
              {isLoading ? (
                <div className="space-y-6 animate-pulse">
                  {/* Header Skeleton */}
                  <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-900 pb-4">
                    <div className="space-y-2">
                      <div className="h-6 w-48 bg-zinc-250 dark:bg-zinc-900 rounded" />
                      <div className="h-3 w-80 bg-zinc-250 dark:bg-zinc-900 rounded" />
                    </div>
                    <div className="h-6 w-32 bg-zinc-250 dark:bg-zinc-900 rounded-full" />
                  </div>

                  {/* Body Content Skeleton */}
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2 h-[320px] bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-900 rounded-2xl p-5 space-y-4">
                      <div className="flex justify-between border-b border-zinc-200 dark:border-zinc-900 pb-3">
                        <div className="h-4 w-36 bg-zinc-250 dark:bg-zinc-900 rounded" />
                        <div className="h-6 w-44 bg-zinc-250 dark:bg-zinc-900 rounded" />
                      </div>
                      <div className="h-48 w-full bg-zinc-50 dark:bg-zinc-950/60 rounded-xl" />
                    </div>

                    <div className="space-y-6">
                      <div className="h-[148px] bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-900 rounded-2xl p-5 space-y-3">
                        <div className="h-4 w-28 bg-zinc-250 dark:bg-zinc-900 rounded" />
                        <div className="h-10 w-full bg-zinc-50 dark:bg-zinc-950/60 rounded-xl" />
                      </div>
                      <div className="h-[148px] bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-900 rounded-2xl p-5 space-y-3">
                        <div className="h-4 w-32 bg-zinc-250 dark:bg-zinc-900 rounded" />
                        <div className="h-12 w-full bg-zinc-50 dark:bg-zinc-950/60 rounded-xl" />
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                /* Actual UI View Component rendering */
                <>
                  {activeView === 'demand' && (
                    <DemandForecastingHub storeInfo={storeInfo} />
                  )}
                  
                  {activeView === 'inventory' && (
                    <InventoryReplenishment 
                      storeInfo={storeInfo} 
                      isMultiStoreMode={isMultiStoreMode} 
                    />
                  )}
                  
                  {activeView === 'financial' && (
                    <FinancialAllocation storeInfo={storeInfo} />
                  )}
                  
                  {activeView === 'transactions' && (
                    <TransactionsLog />
                  )}
                  
                  {activeView === 'settings' && (
                    <Settings theme={theme} setTheme={setTheme} />
                  )}
                </>
              )}

            </div>
          </div>
        </div>
      )}
    </main>
  );
}
