import React from "react";
import { HashRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import RiskExplorer from "./pages/RiskExplorer";
import Customer360 from "./pages/Customer360";
import ModelPerformance from "./pages/ModelPerformance";
import Survival from "./pages/Survival";
import SalesForecast from "./pages/SalesForecast";
import Drift from "./pages/Drift";
import DataQuality from "./pages/DataQuality";
import Datasets from "./pages/Datasets";
import Experiments from "./pages/Experiments";
import { DatasetProvider } from "./context/DatasetContext";

export default function App() {
  return (
    <DatasetProvider>
    <HashRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/risk-explorer" element={<RiskExplorer />} />
          <Route path="/customers/:id" element={<Customer360 />} />
          <Route path="/models" element={<ModelPerformance />} />
          <Route path="/sales-forecast" element={<SalesForecast />} />
          <Route path="/survival" element={<Survival />} />
          <Route path="/drift" element={<Drift />} />
          <Route path="/data-quality" element={<DataQuality />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/experiments" element={<Experiments />} />
        </Routes>
      </Layout>
    </HashRouter>
    </DatasetProvider>
  );
}
