import React, { useMemo, useState } from "react";
import Layout from "../components/Layout";
import Grid from "@mui/material/Grid";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Alert from "@mui/material/Alert";
import LinearProgress from "@mui/material/LinearProgress";
import TextField from "@mui/material/TextField";
import StatCard from "../components/StatCard";
import HotspotMap from "../components/HotspotMap";
import { uploadExcel, trainModel, predictHotspots } from "../api/client";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import PlaceIcon from "@mui/icons-material/Place";
import Divider from "@mui/material/Divider";

function GlassPanel({ title, children, sx }) {
  return (
    <Paper
      elevation={0}
      sx={{
        p: 3,
        border: (theme) => `1px solid ${theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)'}`,
        backgroundColor: (theme) => theme.palette.background.paper,
        backdropFilter: 'blur(8px)',
        boxShadow: (theme) => theme.palette.mode === 'dark' ? '0 6px 18px rgba(0,0,0,0.25)' : '0 6px 18px rgba(0,0,0,0.06)',
        borderRadius: 2,
        ...sx,
      }}
    >
      <Typography variant="h6" sx={{ fontWeight: 800, mb: 1.5 }}>{title}</Typography>
      {children}
    </Paper>
  );
}

export default function DashboardPage() {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const [uploadInfo, setUploadInfo] = useState(null);
  const [trainInfo, setTrainInfo] = useState(null);
  const [hotspotResponse, setHotspotResponse] = useState(null);
  const [topK, setTopK] = useState(5);

  const hotspots = hotspotResponse?.hotspots || [];

  const metrics = useMemo(() => ({
    tripsInserted: uploadInfo?.inserted_trips ?? "—",
    zonesUpserted: uploadInfo?.upserted_zones ?? "—",
    trainLoss: trainInfo?.final_train_loss != null ? Number(trainInfo.final_train_loss).toFixed(4) : "—",
    zones: trainInfo?.num_zones ?? "—"
  }), [uploadInfo, trainInfo]);

  const pickFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setMsg(null);
    setBusy(true);
    try {
      const res = await uploadExcel(file);
      setUploadInfo(res);
      setMsg({ type: "success", text: "Excel uploaded and stored in MongoDB." });
    } catch (err) {
      setMsg({ type: "error", text: err.message });
    } finally {
      setBusy(false);
      e.target.value = "";
    }
  };

  const handleTrain = async () => {
    setMsg(null);
    setBusy(true);
    try {
      const res = await trainModel();
      setTrainInfo(res);
      setMsg({ type: "success", text: "Training completed. Model checkpoint saved." });
    } catch (err) {
      setMsg({ type: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  };

  const handlePredict = async () => {
    setMsg(null);
    setBusy(true);
    try {
      const res = await predictHotspots(Number(topK) || 5);
      setHotspotResponse(res);
      setMsg({ type: "success", text: "Hotspots predicted successfully." });
    } catch (err) {
      setMsg({ type: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Layout
      title="Admin Dashboard"
      rightContent={
        <Stack direction="row" spacing={1.5} alignItems="center">
          <Typography variant="body2" sx={{ opacity: 0.8 }}>API:</Typography>
          <Typography variant="body2" sx={{ fontWeight: 700 }}>{process.env.REACT_APP_API_BASE_URL || "http://127.0.0.1:8000"}</Typography>
        </Stack>
      }
    >
      {busy ? <LinearProgress sx={{ mb: 2.5, borderRadius: 1 }} /> : null}
      {msg ? <Alert severity={msg.type} sx={{ mb: 2.5 }}>{msg.text}</Alert> : null}

      <Grid container spacing={2.5}>
        <Grid item xs={12} md={3}>
          <StatCard
            label="Trips inserted"
            value={metrics.tripsInserted}
            hint="From last Excel upload"
            icon={UploadFileIcon}
            accent="primary"
            sparkline={[8, 12, 10, 14, 18, 16, 22, 20]}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <StatCard
            label="Zones upserted"
            value={metrics.zonesUpserted}
            hint="From last Excel upload"
            icon={PlaceIcon}
            accent="primary"
            sparkline={[4, 6, 5, 7, 9, 10, 9, 12]}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <StatCard
            label="Zones (model)"
            value={metrics.zones}
            hint="Detected zones during training"
            icon={PlaceIcon}
            accent="primary"
            sparkline={[10, 9, 11, 12, 12, 13, 14, 15]}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <StatCard
            label="Final train loss"
            value={metrics.trainLoss}
            hint="Lower is better"
            icon={ModelTrainingIcon}
            accent="primary"
            sparkline={[22, 20, 18, 16, 14, 13, 12, 12]}
          />
        </Grid>

        <Grid item xs={12} md={7}>
          <GlassPanel title="1) Upload Excel to MongoDB">
            <Typography variant="body2" sx={{ opacity: 0.85, mb: 2 }}>
              Upload an Excel file with sheets <b>trips</b> and <b>zones</b>. The backend stores them in MongoDB.
            </Typography>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems="center">
              <Button variant="contained" component="label" startIcon={<UploadFileIcon />} disabled={busy} sx={{ fontWeight: 800 }}>
                Choose Excel (.xlsx)
                <input hidden type="file" accept=".xlsx,.xls" onChange={pickFile} />
              </Button>
              <Typography variant="caption" sx={{ opacity: 0.75 }}></Typography>
            </Stack>
          </GlassPanel>

          <GlassPanel title="2) Train ST-GNN model" sx={{ mt: 2 }}>
            <Typography variant="body2" sx={{ opacity: 0.85, mb: 2 }}>
              Runs preprocessing + aggregation, graph construction, model training, and saves a checkpoint on the backend.
            </Typography>
            <Button variant="outlined" startIcon={<ModelTrainingIcon />} onClick={handleTrain} disabled={busy} sx={{ fontWeight: 800 }}>Train Model</Button>
          </GlassPanel>
        </Grid>

        <Grid item xs={12} md={5}>
          <GlassPanel title="Recent Activity">
            <Typography variant="body2" sx={{ mb: 1 }}>Latest operations and quick details.</Typography>
            <Divider sx={{ mb: 1.5, opacity: 0.08 }} />
            <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Uploads</Typography>
            <Typography variant="body2" sx={{ opacity: 0.85, mb: 1 }}>{uploadInfo ? `Inserted ${uploadInfo.inserted_trips ?? 0} trips — Zones ${uploadInfo.upserted_zones ?? 0}` : "No recent upload"}</Typography>
            <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Training</Typography>
            <Typography variant="body2" sx={{ opacity: 0.85, mb: 1 }}>{trainInfo ? `Checkpoint saved — final loss ${trainInfo?.final_train_loss ?? "—"}` : "No training run"}</Typography>
            <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Predictions</Typography>
            <Typography variant="body2" sx={{ opacity: 0.85 }}>{hotspots?.length ? `${hotspots.length} hotspots returned` : "No predictions yet"}</Typography>
          </GlassPanel>
        </Grid>

        <Grid item xs={12}>
          <GlassPanel title="3) Predict hotspots (Lat/Lng) and view on Google Maps">
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems="center" sx={{ mb: 2 }}>
              <TextField label="Top K" type="number" value={topK} onChange={(e) => setTopK(e.target.value)} inputProps={{ min: 1, max: 50 }} sx={{ width: 140 }} />
              <Button variant="contained" startIcon={<PlaceIcon />} onClick={handlePredict} disabled={busy} sx={{ fontWeight: 800 }}>Predict Hotspots</Button>
              {hotspotResponse?.forecast_for_timebin_start ? (
                <Typography variant="body2" sx={{ opacity: 0.85 }}>Forecast time bin: <b>{new Date(hotspotResponse.forecast_for_timebin_start).toLocaleString()}</b></Typography>
              ) : null}
            </Stack>

            <HotspotMap hotspots={hotspots} />

            {hotspots?.length ? (
              <Paper elevation={0} sx={{ mt: 2, p: 2, border: (theme) => `1px dashed ${theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.12)'}`, backgroundColor: (theme) => theme.palette.background.paper }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>Returned coordinates (for Google Maps markers)</Typography>
                <Divider sx={{ mb: 1.5, opacity: 0.15 }} />
                <pre style={{ margin: 0, overflowX: "auto", opacity: 0.9 }}>{JSON.stringify(hotspots, null, 2)}</pre>
              </Paper>
            ) : null}
          </GlassPanel>
        </Grid>
      </Grid>
    </Layout>
  );
}
