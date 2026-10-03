import React, { useState, useRef, useMemo } from 'react';
import Papa from 'papaparse';
import { saveAs } from 'file-saver';
import { Bar, Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { RandomForestRegression } from 'ml-random-forest';

ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, Title, Tooltip, Legend);

type Row = Record<string, any>;

/* ===== helpers ===== */
const isNumericValue = (v: any) => v !== '' && v != null && !isNaN(Number(v));

const App: React.FC = () => {
  /* ---------- state ---------- */
  const [rawFile, setRawFile] = useState<File | null>(null);
  const [rawData, setRawData] = useState<Row[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [cleanedData, setCleanedData] = useState<Row[]>([]);
  const [cleanedCols, setCleanedCols] = useState<string[]>([]);
  const [changes, setChanges] = useState<string[]>([]);
  const [step, setStep] = useState<number>(0); // 0=upload,1=preview,2=cleaned,3=viz,4=train,5=result
  const [dragOver, setDragOver] = useState(false);

  // viz
  const [barCol, setBarCol] = useState('');
  const [scX, setScX] = useState('');
  const [scY, setScY] = useState('');

  // train
  const [targetCol, setTargetCol] = useState('');
  const [nTrees, setNTrees] = useState(50);
  const [isTraining, setIsTraining] = useState(false);
  const [trainError, setTrainError] = useState('');

  // results
  const [model, setModel] = useState<any>(null);
  const [r2, setR2] = useState<number | null>(null);
  const [taskType, setTaskType] = useState('');
  const [trainRows, setTrainRows] = useState(0);
  const [testRows, setTestRows] = useState(0);
  const [featureImp, setFeatureImp] = useState<[string, number][]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);

  /* ---------- file handling ---------- */
  const handleFile = (file: File) => {
    setRawFile(file);
    Papa.parse<Row>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (res) => {
        setRawData(res.data);
        setColumns(res.meta.fields ?? Object.keys(res.data[0] ?? {}));
      },
    });
  };

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0]);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  /* ---------- step helpers ---------- */
  const stepNames = ['Upload', 'Preview', 'Clean', 'Visualize', 'Train'];
  const currentStepIndex = step <= 1 ? step : step === 2 ? 2 : step === 3 ? 3 : step >= 4 ? 4 : 0;

  /* ---------- preview tables ---------- */
  const head10 = rawData.slice(0, 10);
  const tail10 = rawData.slice(-10);

  const renderValue = (v: any) => {
    if (v === '' || v == null || v === 'NaN' || (typeof v === 'number' && isNaN(v))) {
      return <span className="nan">NaN</span>;
    }
    return String(v);
  };

  const renderTable = (rows: Row[], cols: string[], startIdx = 0) => (
    <div className="table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            <th>#</th>
            {cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              <td style={{ color: 'var(--text-muted)' }}>{startIdx + i}</td>
              {cols.map((c) => (
                <td key={c}>{renderValue(row[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  /* ---------- cleaning pipeline ---------- */
  const cleanData = () => {
    const log: string[] = [];
    let data: Row[] = JSON.parse(JSON.stringify(rawData));
    const allCols = [...columns];

    // Helper: test if value is null / NaN / missing
    const isMissing = (v: any): boolean => {
      if (v === null || v === undefined) return true;
      if (typeof v === 'number' && isNaN(v)) return true;
      if (typeof v === 'string') {
        const s = v.trim().toLowerCase();
        return s === '' || s === 'nan' || s === 'null' || s === 'na' || s === 'n/a' || s === 'none' || s === '?' || s === 'empty' || s === 'undefined';
      }
      return false;
    };

    // 1. Trim whitespace & standardise missing markers across all columns
    data = data.map((row) => {
      const nr: Row = {};
      allCols.forEach((col) => {
        const v = row[col];
        if (isMissing(v)) {
          nr[col] = null;
        } else if (typeof v === 'string') {
          nr[col] = v.trim();
        } else {
          nr[col] = v;
        }
      });
      return nr;
    });
    log.push(`Trimmed whitespace and standardised missing markers (NA, null, ?, empty) across ${allCols.length} columns`);

    // 2. Remove completely empty rows (all fields are null)
    const beforeEmpty = data.length;
    data = data.filter((row) => allCols.some((c) => row[c] !== null));
    const removedEmpty = beforeEmpty - data.length;
    if (removedEmpty) log.push(`Removed ${removedEmpty} completely empty rows`);

    // 3. Drop mostly-empty columns (>60% missing values)
    const total = data.length;
    const colsToDrop: string[] = [];
    allCols.forEach((col) => {
      const missingCount = data.filter((r) => r[col] === null).length;
      if (total > 0 && missingCount / total > 0.6) {
        colsToDrop.push(col);
      }
    });

    if (colsToDrop.length) {
      data = data.map((r) => {
        const nr = { ...r };
        colsToDrop.forEach((c) => delete nr[c]);
        return nr;
      });
      log.push(`Dropped columns exceeding 60% missing: ${colsToDrop.join(', ')}`);
    } else {
      log.push('No columns exceeded 60% missing values');
    }

    const activeCols = allCols.filter((c) => !colsToDrop.includes(c));

    // 4. Remove duplicate rows
    const beforeDup = data.length;
    const seen = new Set<string>();
    data = data.filter((r) => {
      const key = JSON.stringify(r);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    const removedDup = beforeDup - data.length;
    if (removedDup) log.push(`Removed ${removedDup} duplicate rows`);

    // 5. Detect numeric vs categorical columns & convert numeric text
    const numericCols: string[] = [];
    const catCols: string[] = [];

    activeCols.forEach((col) => {
      const nonNullVals = data.map((r) => r[col]).filter((v) => v !== null);
      if (nonNullVals.length === 0) {
        // Fallback default numeric
        numericCols.push(col);
        return;
      }
      const numValidCount = nonNullVals.filter((v) => {
        if (typeof v === 'number') return !isNaN(v);
        const parsed = Number(v);
        return !isNaN(parsed) && String(v).trim() !== '';
      }).length;

      // If at least 70% of non-null values are numeric, treat column as numeric
      if (numValidCount / nonNullVals.length >= 0.7) {
        numericCols.push(col);
        data.forEach((r) => {
          if (r[col] !== null) {
            const num = Number(r[col]);
            r[col] = isNaN(num) ? null : num;
          }
        });
      } else {
        catCols.push(col);
        data.forEach((r) => {
          if (r[col] !== null) {
            r[col] = String(r[col]).trim();
          }
        });
      }
    });
    log.push(`Detected ${numericCols.length} numeric and ${catCols.length} categorical columns; converted numeric text to numbers`);

    // 6. Fill missing values (median for numeric, mode for categorical)
    let filledCount = 0;
    numericCols.forEach((col) => {
      const validNumbers = data
        .map((r) => r[col])
        .filter((v) => typeof v === 'number' && !isNaN(v)) as number[];

      let median = 0;
      if (validNumbers.length > 0) {
        validNumbers.sort((a, b) => a - b);
        const mid = Math.floor(validNumbers.length / 2);
        median = validNumbers.length % 2 === 0
          ? (validNumbers[mid - 1] + validNumbers[mid]) / 2
          : validNumbers[mid];
      }

      data.forEach((r) => {
        if (r[col] === null || r[col] === undefined || (typeof r[col] === 'number' && isNaN(r[col]))) {
          r[col] = median;
          filledCount++;
        }
      });
    });

    catCols.forEach((col) => {
      const validStrings = data
        .map((r) => r[col])
        .filter((v) => v !== null && v !== undefined && v !== '') as string[];

      let mode = 'Unknown';
      if (validStrings.length > 0) {
        const freq: Record<string, number> = {};
        validStrings.forEach((s) => { freq[s] = (freq[s] || 0) + 1; });
        const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1]);
        if (sorted[0]) mode = sorted[0][0];
      }

      data.forEach((r) => {
        if (r[col] === null || r[col] === undefined || r[col] === '') {
          r[col] = mode;
          filledCount++;
        }
      });
    });

    if (filledCount) log.push(`Filled ${filledCount} missing values (median for numeric, most frequent for categorical)`);

    // 7. Cap outliers using IQR 1.5x rule on numeric columns
    let cappedCount = 0;
    numericCols.forEach((col) => {
      const vals = data
        .map((r) => Number(r[col]))
        .filter((v) => !isNaN(v))
        .sort((a, b) => a - b);

      if (vals.length >= 4) {
        const q1 = vals[Math.floor(vals.length * 0.25)];
        const q3 = vals[Math.floor(vals.length * 0.75)];
        const iqr = q3 - q1;
        const lo = q1 - 1.5 * iqr;
        const hi = q3 + 1.5 * iqr;

        data.forEach((r) => {
          const v = Number(r[col]);
          if (!isNaN(v)) {
            if (v < lo) {
              r[col] = Math.round(lo * 1000) / 1000;
              cappedCount++;
            } else if (v > hi) {
              r[col] = Math.round(hi * 1000) / 1000;
              cappedCount++;
            }
          }
        });
      }
    });
    if (cappedCount) log.push(`Capped ${cappedCount} outlier values using the IQR (1.5×) rule`);

    // Final safety guarantee: zero remaining null / undefined / NaN
    activeCols.forEach((col) => {
      const isNum = numericCols.includes(col);
      data.forEach((r) => {
        if (r[col] === null || r[col] === undefined || (typeof r[col] === 'number' && isNaN(r[col]))) {
          r[col] = isNum ? 0 : 'N/A';
        }
      });
    });

    log.push(`Final dataset: ${data.length} rows × ${activeCols.length} columns`);

    setCleanedData(data);
    setCleanedCols(activeCols);
    setChanges(log);
    setStep(2);
  };

  const downloadCSV = () => {
    const csv = Papa.unparse(cleanedData);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, 'cleaned_data.csv');
  };

  /* ---------- chart data ---------- */
  const darkChartOptions = (title: string): any => ({
    responsive: true,
    plugins: {
      legend: { display: false },
      title: { display: false },
    },
    scales: {
      x: { ticks: { color: '#8899a6', font: { size: 10 } }, grid: { color: '#253341' } },
      y: { ticks: { color: '#8899a6', font: { size: 10 } }, grid: { color: '#253341' } },
    },
  });

  const barChartData = useMemo(() => {
    if (!barCol || !cleanedData.length) return null;
    const counts: Record<string, number> = {};
    cleanedData.forEach((r) => { const v = String(r[barCol]); counts[v] = (counts[v] || 0) + 1; });
    return {
      labels: Object.keys(counts),
      datasets: [{ label: barCol, data: Object.values(counts), backgroundColor: '#10b981' }],
    };
  }, [barCol, cleanedData]);

  const scatterChartData = useMemo(() => {
    if (!scX || !scY || !cleanedData.length) return null;
    return {
      datasets: [{
        label: `${scX} vs ${scY}`,
        data: cleanedData.map((r) => ({ x: Number(r[scX]), y: Number(r[scY]) })),
        backgroundColor: '#f59e0b',
        pointRadius: 5,
      }],
    };
  }, [scX, scY, cleanedData]);

  /* ---------- training ---------- */
  const numericCleanedCols = useMemo(
    () => cleanedCols.filter((c) => cleanedData.length && cleanedData.every((r) => isNumericValue(r[c]))),
    [cleanedCols, cleanedData]
  );

  const trainModel = () => {
    if (!targetCol) return;
    setIsTraining(true);
    setTrainError('');

    // use setTimeout so the UI updates to show "Training..."
    setTimeout(() => {
      try {
        const featureCols = cleanedCols.filter((c) => c !== targetCol);

        // Label-encode any categorical feature columns
        const encoders: Record<string, Record<string, number>> = {};
        featureCols.forEach((col) => {
          if (!cleanedData.every((r) => typeof r[col] === 'number' && !isNaN(r[col]))) {
            const uniq = Array.from(new Set(cleanedData.map((r) => String(r[col]))));
            encoders[col] = {};
            uniq.forEach((v, i) => (encoders[col][v] = i));
          }
        });

        // Check if target is categorical or numeric
        const targetIsCategorical = !cleanedData.every((r) => typeof r[targetCol] === 'number' && !isNaN(r[targetCol]));
        const targetEncoder: Record<string, number> = {};
        if (targetIsCategorical) {
          const uniqTarget = Array.from(new Set(cleanedData.map((r) => String(r[targetCol]))));
          uniqTarget.forEach((v, i) => (targetEncoder[v] = i));
        }

        const allRows = cleanedData.map((row) => {
          const features = featureCols.map((c) => {
            if (encoders[c]) return encoders[c][String(row[c])] ?? 0;
            const n = Number(row[c]);
            return isNaN(n) ? 0 : n;
          });
          const target = targetIsCategorical
            ? (targetEncoder[String(row[targetCol])] ?? 0)
            : Number(row[targetCol]);
          return { features, target: isNaN(target) ? 0 : target };
        });

        // Train-test split 80/20
        const shuffled = [...allRows].sort(() => Math.random() - 0.5);
        const splitIdx = Math.floor(shuffled.length * 0.8);
        const trainSet = shuffled.slice(0, splitIdx);
        const testSet = shuffled.slice(splitIdx);

        const Xtrain = trainSet.map((r) => r.features);
        const ytrain = trainSet.map((r) => r.target);
        const Xtest = testSet.map((r) => r.features);
        const ytest = testSet.map((r) => r.target);

        // Detect task type
        const uniqueTargets = new Set(ytrain);
        const isClassification = uniqueTargets.size <= 20;

        const maxFeat = Math.max(1, Math.floor(Math.sqrt(featureCols.length)));
        const rf = new RandomForestRegression({ nEstimators: nTrees, maxFeatures: maxFeat, seed: 42 });
        rf.train(Xtrain, ytrain);
        setModel(rf);

        // Evaluate
        const preds = rf.predict(Xtest);

        if (isClassification) {
          // Accuracy (round predictions to nearest class)
          const classes = Array.from(uniqueTargets);
          let correct = 0;
          preds.forEach((p: number, i: number) => {
            const nearest = classes.reduce((best, c) => (Math.abs(p - c) < Math.abs(p - best) ? c : best), classes[0]);
            if (nearest === ytest[i]) correct++;
          });
          setR2(correct / ytest.length);
          setTaskType('Classification');
        } else {
          const meanY = ytest.reduce((a, b) => a + b, 0) / ytest.length;
          const ssRes = ytest.reduce((s, v, i) => s + (v - preds[i]) ** 2, 0);
          const ssTot = ytest.reduce((s, v) => s + (v - meanY) ** 2, 0);
          setR2(1 - ssRes / ssTot);
          setTaskType('Regression');
        }

        setTrainRows(trainSet.length);
        setTestRows(testSet.length);

        // Feature importances via permutation: shuffle each feature and measure accuracy drop
        const basePreds = rf.predict(Xtest);
        const baseMSE = ytest.reduce((s, v, i) => s + (v - basePreds[i]) ** 2, 0) / ytest.length;

        const imp: number[] = featureCols.map((_, fIdx) => {
          // Create a copy with feature fIdx shuffled
          const Xshuffled = Xtest.map((row) => [...row]);
          // Fisher-Yates shuffle for column fIdx
          for (let i = Xshuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            const tmp = Xshuffled[i][fIdx];
            Xshuffled[i][fIdx] = Xshuffled[j][fIdx];
            Xshuffled[j][fIdx] = tmp;
          }
          const shuffPreds = rf.predict(Xshuffled);
          const shuffMSE = ytest.reduce((s, v, i) => s + (v - shuffPreds[i]) ** 2, 0) / ytest.length;
          return Math.max(0, shuffMSE - baseMSE);
        });

        // Normalise so they sum to 1
        const impSum = imp.reduce((a, b) => a + b, 0) || 1;
        const pairs: [string, number][] = featureCols.map((c, i) => [c, imp[i] / impSum]);
        pairs.sort((a, b) => b[1] - a[1]);
        setFeatureImp(pairs);
        setStep(5);
      } catch (e: any) {
        setTrainError(e.message || 'Training failed');
      } finally {
        setIsTraining(false);
      }
    }, 50);
  };

  const downloadModel = () => {
    if (!model) return;
    const blob = new Blob([JSON.stringify((model as any).toJSON())], { type: 'application/json' });
    saveAs(blob, 'random_forest_model.json');
  };

  /* ============================= RENDER ============================= */
  return (
    <div className="app-container">
      {/* HEADER */}
      <p className="brand">DATAFORGE</p>
      <h1 className="hero-title">From raw CSV to trained model.</h1>
      <p className="hero-subtitle">
        Upload a dataset, preview it, clean it automatically, visualize it and train a Random Forest you can download.
      </p>

      {/* STEP PILLS */}
      <div className="step-pills">
        {stepNames.map((name, idx) => (
          <span
            key={name}
            className={`step-pill ${currentStepIndex > idx ? 'done' : ''} ${currentStepIndex === idx ? 'active' : ''}`}
          >
            {currentStepIndex > idx ? <span className="check">✓</span> : <span style={{ opacity: 0.5 }}>{idx + 1}</span>}
            {name}
          </span>
        ))}
      </div>

      {/* TOP BUTTONS */}
      <div className="top-actions">
        <button
          className="btn btn-primary"
          onClick={() => { setStep(0); setRawFile(null); setRawData([]); setCleanedData([]); setModel(null); }}
        >
          ↑ Upload Dataset
        </button>
        <button
          className="btn btn-next"
          disabled={!rawFile || step > 0}
          onClick={() => setStep(1)}
        >
          Next →
        </button>
      </div>

      {/* ============ STEP 0 – UPLOAD ============ */}
      {step === 0 && (
        <div className="card">
          <div className="card-title"><span className="icon">↑</span> Upload your dataset</div>
          <div
            className={`dropzone ${dragOver ? 'drag-over' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            <div className="dropzone-icon">↑</div>
            <div className="dropzone-text">Drop a CSV file here or click to browse</div>
            <div className="dropzone-hint">Works best with 1000+ rows</div>
            <input type="file" accept=".csv" onChange={onFileInput} ref={fileInputRef} />
          </div>
          {rawFile && (
            <div className="file-info">
              <span className="check">✓</span> {rawFile.name} — {rawData.length} rows × {columns.length} columns. Click <b>Next</b>.
            </div>
          )}
        </div>
      )}

      {/* ============ STEP 1 – PREVIEW ============ */}
      {step === 1 && (
        <div className="card">
          <div className="card-title"><span className="icon">→</span> Dataset preview</div>
          <div className="table-label">head(10)</div>
          {renderTable(head10, columns, 0)}
          <div className="table-label" style={{ marginTop: 18 }}>tail(10)</div>
          {renderTable(tail10, columns, rawData.length - 10)}
          <div className="action-row">
            <button className="btn btn-secondary" onClick={cleanData}>✨ Clean Data</button>
          </div>
        </div>
      )}

      {/* ============ STEP 2 – CLEANED ============ */}
      {step === 2 && (
        <>
          <div className="card">
            <div className="card-title"><span className="icon">✨</span> Cleaned data</div>
            <ul className="clean-log">
              {changes.map((c, i) => (
                <li key={i}><span className="check">✓</span> {c}</li>
              ))}
            </ul>
            <div style={{ marginTop: 18 }}>
              {renderTable(cleanedData.slice(0, 10), cleanedCols, 0)}
            </div>
            <div className="action-row">
              <button className="btn btn-primary" onClick={() => setStep(3)}>📊 Visualize</button>
              <button className="btn btn-secondary" onClick={downloadCSV}>↓ Download cleaned CSV</button>
            </div>
          </div>
        </>
      )}

      {/* ============ STEP 3 – VISUALIZE ============ */}
      {step === 3 && (
        <>
          <div className="card">
            <div className="card-title"><span className="icon">📊</span> Visualization</div>

            {/* Bar chart */}
            <div className="select-row">
              <label>Distribution of</label>
              <select className="dark-select" value={barCol} onChange={(e) => setBarCol(e.target.value)}>
                <option value="">—select—</option>
                {cleanedCols.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            {barChartData && (
              <div className="chart-container">
                <Bar data={barChartData} options={darkChartOptions(barCol)} />
              </div>
            )}

            {/* Scatter plot */}
            <div className="select-row" style={{ marginTop: 18 }}>
              <select className="dark-select" value={scX} onChange={(e) => setScX(e.target.value)}>
                <option value="">X axis</option>
                {cleanedCols.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
              <span className="scatter-vs">vs</span>
              <select className="dark-select" value={scY} onChange={(e) => setScY(e.target.value)}>
                <option value="">Y axis</option>
                {cleanedCols.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            {scatterChartData && (
              <div className="chart-container">
                <Scatter data={scatterChartData} options={darkChartOptions('')} />
              </div>
            )}

            <div className="action-row">
              <button className="btn btn-primary" onClick={() => setStep(4)}>⚙ Train Model</button>
            </div>
          </div>
        </>
      )}

      {/* ============ STEP 4 – TRAIN ============ */}
      {step === 4 && (
        <div className="train-card">
          <div className="card-title"><span className="icon">⚙</span> Train Random Forest</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 14 }}>
            Target column (what to predict)
          </div>
          <div className="train-row">
            <div className="train-field">
              <select className="dark-select" value={targetCol} onChange={(e) => setTargetCol(e.target.value)}>
                <option value="">—select target—</option>
                {cleanedCols.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="train-field">
              <label>Number of trees: {nTrees}</label>
              <div className="range-row">
                <input type="range" min={10} max={300} value={nTrees} onChange={(e) => setNTrees(Number(e.target.value))} />
              </div>
            </div>
          </div>
          <button
            className="btn btn-primary"
            disabled={!targetCol || isTraining}
            onClick={trainModel}
          >
            {isTraining ? <><span className="spinner" /> Training...</> : '⚙ Train Random Forest'}
          </button>
          {trainError && <p className="error-msg">{trainError}</p>}
        </div>
      )}

      {/* ============ STEP 5 – RESULTS ============ */}
      {step === 5 && model && (
        <div className="card">
          <div className="card-title"><span className="icon">⚙</span> Trained model</div>

          <div className="result-grid">
            <div className="result-card">
              <div className="label">Task</div>
              <div className="value white" style={{ fontSize: 16 }}>{taskType}</div>
            </div>
            <div className="result-card">
              <div className="label">Accuracy</div>
              <div className="value">{((r2 ?? 0) * 100).toFixed(1)}%</div>
            </div>
            <div className="result-card">
              <div className="label">Train rows</div>
              <div className="value white">{trainRows}</div>
            </div>
            <div className="result-card">
              <div className="label">Test rows</div>
              <div className="value white">{testRows}</div>
            </div>
          </div>

          <div className="fi-title">Feature importance</div>
          <ul className="fi-list">
            {featureImp.map(([name, val]) => {
              const maxVal = featureImp[0]?.[1] || 1;
              const pct = (val / maxVal) * 100;
              return (
                <li key={name} className="fi-item">
                  <span className="fi-label">{name}</span>
                  <div className="fi-bar-bg">
                    <div className="fi-bar" style={{ width: `${pct}%` }} />
                  </div>
                </li>
              );
            })}
          </ul>

          <div className="action-row" style={{ marginTop: 20 }}>
            <button className="btn btn-coral" onClick={downloadModel}>↓ Download trained model</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
