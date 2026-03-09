// --- USUNIĘTO SZTYWNĄ REDUKCJĘ WYSOKOŚCI W PIKSELACH ---
// Od teraz za wysokość i responsywność obu wykresów odpowiada CSS (patrz krok 2)

const priceDOM = document.getElementById('chart-price');
const pnlDOM = document.getElementById('chart-pnl');
const priceChart = echarts.init(priceDOM, 'dark');
const pnlChart = echarts.init(pnlDOM, 'dark');

let currentStatus = "STARTUP";
let currentPriceData =[]; 
let lastMinuteTracker = new Date().getMinutes();
let lastPnlLength = 0; // <--- DODAJ TĘ LINIJKĘ
let lastKnownCandleTs = 0; // Śledzimy czas z serwera, nie z systemu operacyjnego.

// Zmienne śledzące, który punkt na osi X jest obecnie pierwszym widocznym z lewej strony
let currentZoomStartPrice = 0;
let currentZoomStartPnl = 0;

// Nasłuchujemy przesuwania i przybliżania wykresów, by zawsze wiedzieć gdzie jest lewa krawędź
priceChart.on('dataZoom', function () {
    let opt = priceChart.getOption();
    if(opt && opt.dataZoom && opt.dataZoom.length > 0) {
        currentZoomStartPrice = Math.max(0, Math.floor(opt.dataZoom[0].startValue || 0));
    }
});
pnlChart.on('dataZoom', function () {
    let opt = pnlChart.getOption();
    if(opt && opt.dataZoom && opt.dataZoom.length > 0) {
        currentZoomStartPnl = Math.max(0, Math.floor(opt.dataZoom[0].startValue || 0));
    }
});

const formatProb = (p, w) => {
    if (p == null || isNaN(p)) return "--";
    const cls = p > 50.1 ? 'up' : (p < 49.9 ? 'down' : 'neutral');
    const weightHtml = w ? ` <span style="color:#666;font-size:0.8em;">(x${w.toFixed(2)})</span>` : '';
    return `<span class="${cls}">${p.toFixed(1)}%</span>${weightHtml}`;
};

const formatConsensus = (signal, prob, mult) => {
    const s = signal || "NEUTRAL";
    const color = (s === "BUY") ? "#00e676" : (s === "SELL" ? "#ff1744" : "#666");
    const mHtml = mult !== undefined ? ` <span style="color:#666;font-size:0.8em;">(x${mult > 0 ? '1' : '-1'})</span>` : '';
    return `<span style="color:${color};font-weight:900;">${s} ${prob.toFixed(1)}%</span>${mHtml}`;
};

const extractValues = (arr) => {
    if (!arr) return[];
    return arr.map(item => Array.isArray(item) ? item[1] : item);
};

const extractDates = (arr) => {
    if (!arr || arr.length === 0) return[];
    return arr.map(item => {
        if (Array.isArray(item)) {
            const ts = item[0];
            if (typeof ts === 'number') {
                const d = new Date(ts);
                const YYYY = d.getFullYear();
                const MM = String(d.getMonth()+1).padStart(2, '0');
                const DD = String(d.getDate()).padStart(2, '0');
                const HH = String(d.getHours()).padStart(2, '0');
                const mm = String(d.getMinutes()).padStart(2, '0');
                return `${YYYY}-${MM}-${DD} ${HH}:${mm}`;
            }
            return String(ts); 
        }
        return "";
    });
};

const alignPnl = (targetArr, masterArr) => {
    if (!masterArr || masterArr.length === 0) return extractValues(targetArr);
    if (!targetArr || targetArr.length === 0) return new Array(masterArr.length).fill(null);

    const isPairs = Array.isArray(masterArr[0]) && Array.isArray(targetArr[0]);

    if (isPairs) {
        const targetMap = new Map();
        targetArr.forEach(item => targetMap.set(item[0], item[1]));

        let result =[];
        let lastKnownValue = targetArr[0][1]; 

        masterArr.forEach(masterItem => {
            const ts = masterItem[0];
            if (targetMap.has(ts)) {
                lastKnownValue = targetMap.get(ts);
            }
            result.push(lastKnownValue); 
        });
        return result;
    } else {
        let vals = extractValues(targetArr);
        let result =[];
        let lastVal = vals.length > 0 ? vals[vals.length - 1] : null;

        for (let i = 0; i < masterArr.length; i++) {
            if (i < vals.length) {
                result.push(vals[i]);
            } else {
                result.push(lastVal);
            }
        }
        return result;
    }
};

function getOption(isPrice) {
    return {
        backgroundColor: 'transparent',
        animation: false,
        tooltip: { 
            trigger: 'axis', 
            axisPointer: { type: 'cross' },
            confine: true, // <--- 1 ZMIANA: Wymusza trzymanie chmurki w obrębie kontenera
            position: function (point, params, dom, rect, size) {
                // Zabezpieczenie rozmiaru chmurki
                let contentWidth = size.contentSize[0] || 200;
                let contentHeight = size.contentSize[1] || 150;
                
                // Odsunięcie w poziomie od pionowej linii (kursora)
                let xOffset = 60; 
                
                // Szerokość miejsca zajmowanego przez liczby na osi Y z prawej strony (margines bezpieczeństwa)
                let yAxisProtectionZone = 80; 

                // Domyślnie chmurka z prawej strony
                let xPos = point[0] + xOffset;
                
                // Jeśli chmurka wjechałaby w strefę chronioną osi Y lub za ekran -> przerzuć na lewo
                if (xPos + contentWidth > size.viewSize[0] - yAxisProtectionZone) {
                    xPos = point[0] - xOffset - contentWidth;
                }

                // <--- 2 ZMIANA: Zabezpieczenie chmurki przed wyjazdem w lewo poza ekran
                if (xPos < 10) {
                    xPos = 10;
                }

                // Środek chmurki w pionie na wysokości kursora
                let yPos = point[1] - (contentHeight / 2);
                
                // Zabezpieczenie przed ucieczką chmurki za górną/dolną krawędź wykresu
                if (yPos < 10) yPos = 10;
                if (yPos + contentHeight > size.viewSize[1] - 10) {
                    yPos = size.viewSize[1] - contentHeight - 10;
                }

                return [xPos, yPos];
            },

            formatter: function (params) {
                let tooltipHtml = `${params[0].axisValue.replace('\n', ' ')}<br/>`;
                params.forEach(item => {
                    if (item.value !== null && item.value !== undefined && !isNaN(item.value)) {
                        const actualColor = item.color;
                        const customMarker = `<span style="display:inline-block;margin-right:5px;border-radius:50%;width:10px;height:10px;background-color:${actualColor};"></span>`;
                        tooltipHtml += `${customMarker} ${item.seriesName}: <b>${Math.round(item.value)}</b><br/>`;
                    }
                });
                return tooltipHtml;
            }
        },
        legend: { 
            show: true, 
            top: 0, 
            icon: 'rect', 
            itemWidth: 15, 
            itemHeight: 2, 
            textStyle: { color: '#888', fontSize: 10, padding: [0, 5] } 
        },
        grid: { 
            top: 40, 
            left: 15, 
            right: 15, 
            bottom: isPrice ? 40 : 80, 
            containLabel: true 
        },
        xAxis: { 
            type: 'category',
            boundaryGap: false, 
            axisLine: { lineStyle: { color: '#333' } },
            splitLine: { 
                show: true, 
                lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } 
            },
            axisLabel: {
                showMinLabel: true, 
                formatter: function (value, index) {
                    if (!value) return "";
                    
                    let startIdx = isPrice ? currentZoomStartPrice : currentZoomStartPnl;
                    let parts = value.split(' ');
                    let datePart = parts[0]; 
                    let timePart = parts.length > 1 ? parts[1].substring(0, 5) : ""; 

                    if (index === 0 || index === startIdx || index === startIdx + 1) {
                        return `${datePart}\n${timePart}`; 
                    }
                    
                    return timePart;
                }
            }
        },
        yAxis: { 
            type: 'value', 
            position: 'right', 
            scale: true,
            splitLine: { lineStyle: { color: '#111' } },
            min: 'dataMin', 
            max: 'dataMax',
            axisLabel: { formatter: (value) => Math.round(value) } 
        },
        dataZoom:[
            { type: 'inside', filterMode: 'filter' },
            { 
                type: 'slider', 
                show: !isPrice, 
                filterMode: 'filter', 
                bottom: 10, height: 30, borderColor: '#333', 
                fillerColor: 'rgba(0, 230, 118, 0.1)', handleSize: '100%' 
            }
        ],
        series:[]
    };
}

async function updateDashboard() {
    try {
        const r = await fetch('/api/init');
        const d = await r.json();
        if (!d.history) return;

        const m = d.models;

        const mcWin = m.config.win !== undefined ? m.config.win : '--';
        const mcAhead = m.config.ahead !== undefined ? m.config.ahead : '--';
        const mcIter = m.config.iter !== undefined ? m.config.iter : '--';
        const mcT = m.config.t !== undefined ? m.config.t : '--';
        const mcS = m.config.s !== undefined ? m.config.s : '--';
        
        document.getElementById('mc-specs').innerText = `{Win:${mcWin} Ahead:${mcAhead} Iter:${mcIter} T:${mcT} S:${mcS}}`;
        document.getElementById('mc-val').innerHTML = formatProb(m.mc_prob, m.weights ? m.weights.mc : null);
        
        const rfRawVal = (m.rf_raw !== undefined && !isNaN(m.rf_raw)) ? parseFloat(m.rf_raw).toFixed(3) : '--';
        const rfAccVal = (m.rf_acc !== undefined && !isNaN(m.rf_acc)) ? parseFloat(m.rf_acc).toFixed(3) : '--';
        
        document.getElementById('rf-math').innerText = `RF | Raw:${rfRawVal}% Acc:${rfAccVal}`;

        document.getElementById('rf-val').innerHTML = formatProb(m.rf_prob, m.weights ? m.weights.rf : null);

        document.getElementById('kan-val').innerHTML = formatProb(m.kan_prob, m.weights.kan);
        document.getElementById('net-val').innerHTML = formatProb(m.neural_prob, m.weights.net);
        document.getElementById('sim-val').innerHTML = formatProb(m.simdiff_prob, 1.0);
        document.getElementById('hakan-val').innerHTML = formatProb(m.hakan_val, 1.0);

        document.getElementById('final-val').innerHTML = formatConsensus(d.consensus_signal, m.consensus_val, m.mult);
        document.getElementById('sim-final-val').innerHTML = formatConsensus(d.simdiff_signal, m.simdiff_prob, m.simdiff_mult);
        document.getElementById('hakan-final-val').innerHTML = formatConsensus(d.hakan_signal, m.hakan_val, m.hakan_mult);

        const priceOpt = getOption(true);

        const allDates =[...d.dates, ...(d.forecast_dates || [])];
        const fullX = Array.from(new Set(allDates)).sort(); 
        priceOpt.xAxis.data = fullX;

        const alignHistory = (histArr) => fullX.map(date => {
            const idx = d.dates.indexOf(date);
            return idx !== -1 ? histArr[idx] : null;
        });

        const alignForecast = (forecastArr, forcePriceAnchor) => {
            if (!forecastArr || forecastArr.length === 0) return fullX.map(() => null);
            let result = fullX.map(() => null);
            
            d.forecast_dates.forEach((fDate, i) => {
                const xIdx = fullX.indexOf(fDate);
                if (xIdx !== -1) result[xIdx] = forecastArr[i];
            });

            if (forcePriceAnchor && d.forecast_dates.length > 0) {
                const anchorDate = d.forecast_dates[0];
                const histIdx = d.dates.indexOf(anchorDate);
                const xIdx = fullX.indexOf(anchorDate);
                if (histIdx !== -1 && xIdx !== -1 && d.history[histIdx] !== undefined) {
                    result[xIdx] = d.history[histIdx];
                }
            }
            return result;
        };

        currentPriceData = alignHistory(d.history);

        priceOpt.series =[
            { name: 'Price', type: 'line', data: currentPriceData, showSymbol: false, symbol: 'circle', symbolSize: 12, color: '#2979ff', lineStyle: { width: 4 }, z: 10 },
            { name: 'SimNet', type: 'line', data: alignForecast(d.stoch, true), showSymbol: false, symbol: 'circle', symbolSize: 12, color: '#ff1744', lineStyle: { width: 1, opacity: 0.3 } },
            { name: 'Trend', type: 'line', data: alignForecast(d.trend, true), showSymbol: false, symbol: 'circle', symbolSize: 12, color: '#ffea00', lineStyle: { type: 'dashed', width: 2 } },
            { name: 'SimDiff', type: 'line', data: alignForecast(d.simdiff_curve, true), showSymbol: false, symbol: 'circle', symbolSize: 12, color: '#d400ff', lineStyle: { width: 2 } },
            { name: 'HaKAN', type: 'line', data: alignForecast(d.hakan_curve, true), showSymbol: false, symbol: 'circle', symbolSize: 12, color: '#ff9100', lineStyle: { width: 2 } },
            { name: 'Res', type: 'line', data: alignForecast(d.res, true), showSymbol: false, symbol: 'circle', symbolSize: 12, color: '#00e676', lineStyle: { opacity: 0.3, type: 'dotted' } },
            { name: 'Sup', type: 'line', data: alignForecast(d.sup, true), showSymbol: false, symbol: 'circle', symbolSize: 12, color: '#ff1744', lineStyle: { opacity: 0.3, type: 'dotted' } }
        ];

        priceChart.setOption(priceOpt, false); 

        const pnlOpt = getOption(false);
        const pnlDates = extractDates(d.pnl);
        pnlOpt.xAxis.data = pnlDates.length > 0 ? pnlDates : d.dates; 

        pnlOpt.series =[
            { name: 'Main', type: 'line', data: extractValues(d.pnl), showSymbol: false, symbol: 'circle', symbolSize: 12, color: '#00e676' },
            { name: 'SimDiff', type: 'line', data: alignPnl(d.simdiff_pnl, d.pnl), showSymbol: false, symbol: 'circle', symbolSize: 12, color: '#d400ff' },
            { name: 'HaKAN', type: 'line', data: alignPnl(d.hakan_pnl, d.pnl), showSymbol: false, symbol: 'circle', symbolSize: 12, color: '#ff9100' }
        ];

        const currentPnlLength = pnlOpt.xAxis.data.length;

        if (currentPnlLength > lastPnlLength) {
            const startValueIdx = Math.max(0, currentPnlLength - 30); 
            const endValueIdx = currentPnlLength - 1;                

            pnlOpt.dataZoom[0].startValue = startValueIdx;
            pnlOpt.dataZoom[0].endValue = endValueIdx;
            pnlOpt.dataZoom[1].startValue = startValueIdx;
            pnlOpt.dataZoom[1].endValue = endValueIdx;
            
            lastPnlLength = currentPnlLength;
            currentZoomStartPnl = startValueIdx; 

        } else {
            delete pnlOpt.dataZoom;
        }

        pnlChart.setOption(pnlOpt, false); 

    } catch (e) { console.error("Update Error:", e); }
}

setInterval(async () => {
    try {
        const r = await fetch('/api/current_price');
        const d = await r.json();
        
        document.getElementById('last-price').innerText = d.price ? d.price.toFixed(0) : "--";
        document.getElementById('server-status').innerText = d.status;
        
        const loader = document.getElementById('loading-overlay');
        if(loader) loader.style.visibility = (d.status === "TRAINING") ? "visible" : "hidden";

        const firstNullIdx = currentPriceData.findIndex(val => val === null || val === undefined);
        const activeIdx = firstNullIdx !== -1 ? firstNullIdx - 1 : currentPriceData.length - 1;

        if (activeIdx >= 0 && currentPriceData[activeIdx] !== null) {
            currentPriceData[activeIdx] = d.price;
            priceChart.setOption({ series: [{ name: 'Price', data: currentPriceData }] });
        }

        if (d.closed_candle && d.closed_candle.ts) {
            if (lastKnownCandleTs === 0) {
                lastKnownCandleTs = d.closed_candle.ts;
            } else if (d.closed_candle.ts > lastKnownCandleTs) {
                lastKnownCandleTs = d.closed_candle.ts;
                await updateDashboard();
            }
        }

        if (currentStatus === "TRAINING" && d.status === "READY") {
            await updateDashboard();
        }
        currentStatus = d.status;
    } catch (e) {}
}, 3000);

window.addEventListener('resize', () => { 
    if(priceChart) priceChart.resize(); 
    if(pnlChart) pnlChart.resize(); 
});

updateDashboard();