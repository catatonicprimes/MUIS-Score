import { useState } from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import { Activity, MapPin, Search, AlertCircle, Database, Crosshair } from 'lucide-react';
import L from 'leaflet';

// Fix for default marker icon in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Component to dynamically update map center
function ChangeView({ center, zoom }) {
  const map = useMap();
  map.setView(center, zoom);
  return null;
}

function App() {
  const [formData, setFormData] = useState({ neighbourhood: '', city: '', country: '' });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.neighbourhood || !formData.city || !formData.country) {
      setError("Please fill all fields.");
      return;
    }
    setError('');
    setLoading(true);
    setResult(null);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Prediction failed");
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getRadarData = (features) => {
    if (!features) return [];
    return [
      { subject: 'Density', A: (features.F01_poi_density + features.F06_residential_density) / 2, fullMark: 100 },
      { subject: 'Diversity', A: (features.F02_land_use_entropy + features.F07_housing_diversity) * 30, fullMark: 100 },
      { subject: 'Mobility', A: (features.F13_pedestrian_count + features.F16_transit_access + features.F22_intersection_density) * 2, fullMark: 100 },
      { subject: 'Economy', A: (features.F14_shop_variety + features.F15_night_economy + features.F09_street_cafe_density) * 10, fullMark: 100 },
      { subject: 'Green', A: (features.F30_green_space_ratio) * 100, fullMark: 100 },
    ].map(d => ({...d, A: Math.min(100, Math.max(0, isNaN(d.A) ? 0 : d.A))}));
  };

  return (
    <div className="min-h-screen bg-[#F5F5F7] text-[#1D1D1F] selection:bg-primary/20 py-12 px-4 sm:px-6 lg:px-8 font-sans antialiased relative overflow-hidden">
      
      {/* Decorative blurred background blobs to enhance glassmorphism */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-blue-100/50 blur-3xl -z-10 pointer-events-none"></div>
      <div className="absolute bottom-[-10%] right-[-10%] w-[30%] h-[30%] rounded-full bg-green-100/40 blur-3xl -z-10 pointer-events-none"></div>

      {/* Header / Search Top Bar */}
      <div className="max-w-7xl mx-auto mb-10 animate-fade-in-up">
        <div className="flex flex-col lg:flex-row justify-between items-center gap-8 bg-white/70 backdrop-blur-xl border border-white/40 rounded-3xl p-6 md:p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)]">
          <div className="text-center lg:text-left">
            <h1 className="text-3xl font-bold tracking-tight flex items-center justify-center lg:justify-start gap-3 text-gray-900">
              <div className="bg-primary/10 p-2 rounded-2xl">
                <Database className="h-7 w-7 text-primary" />
              </div>
              MUIS <span className="text-gray-400 font-medium">V2.0</span>
            </h1>
            <p className="text-gray-500 text-sm mt-2 flex items-center justify-center lg:justify-start gap-1.5 font-medium">
              <Crosshair className="h-4 w-4" /> Predictive Analytics Engine
            </p>
          </div>

          <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-4 w-full lg:w-auto">
            <input 
              type="text" name="neighbourhood" value={formData.neighbourhood} onChange={handleChange}
              className="bg-white/50 border border-gray-200/60 text-gray-900 placeholder-gray-400 rounded-full px-6 py-3.5 focus:outline-none focus:border-primary/50 focus:ring-4 focus:ring-primary/10 focus:bg-white shadow-sm transition-all text-sm font-medium w-full sm:w-40 md:w-48"
              placeholder="Neighbourhood"
            />
            <input 
              type="text" name="city" value={formData.city} onChange={handleChange}
              className="bg-white/50 border border-gray-200/60 text-gray-900 placeholder-gray-400 rounded-full px-6 py-3.5 focus:outline-none focus:border-primary/50 focus:ring-4 focus:ring-primary/10 focus:bg-white shadow-sm transition-all text-sm font-medium w-full sm:w-40 md:w-48"
              placeholder="City"
            />
            <input 
              type="text" name="country" value={formData.country} onChange={handleChange}
              className="bg-white/50 border border-gray-200/60 text-gray-900 placeholder-gray-400 rounded-full px-6 py-3.5 focus:outline-none focus:border-primary/50 focus:ring-4 focus:ring-primary/10 focus:bg-white shadow-sm transition-all text-sm font-medium w-full sm:w-40 md:w-48"
              placeholder="Country"
            />
            <button 
              type="submit" disabled={loading}
              className="bg-primary hover:bg-primaryDark text-white font-medium px-8 py-3.5 rounded-full transition-all shadow-md hover:shadow-lg hover:-translate-y-0.5 active:translate-y-0 active:shadow-sm flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
            >
              <Search className="h-5 w-5" />
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          </form>
        </div>
        
        {error && (
          <div className="mt-6 mx-auto max-w-lg bg-red-50 border border-red-100 text-red-600 rounded-2xl p-4 flex items-center justify-center gap-3 text-sm font-medium shadow-sm animate-fade-in-up">
            <AlertCircle className="h-5 w-5 flex-shrink-0" /> 
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Sleek Skeleton Loading */}
      {loading && (
        <div className="max-w-7xl mx-auto py-24 animate-fade-in-up">
          <div className="flex flex-col items-center justify-center space-y-8">
            <div className="relative">
              <div className="w-16 h-16 border-4 border-gray-200/60 rounded-full"></div>
              <div className="w-16 h-16 border-4 border-primary rounded-full border-t-transparent animate-spin absolute top-0 left-0"></div>
            </div>
            <div className="text-gray-500 font-medium tracking-tight animate-pulse-subtle">
              Processing spatial telemetry...
            </div>
          </div>
        </div>
      )}

      {/* Bento Grid Results */}
      {result && !loading && (
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 lg:grid-cols-12 gap-6 md:gap-8">
          
          {/* Main Score Card (Span 4) */}
          <div className="col-span-1 md:col-span-2 lg:col-span-4 bg-white/80 backdrop-blur-xl border border-white/40 rounded-[2rem] p-8 md:p-10 flex flex-col justify-between shadow-[0_8px_30px_rgb(0,0,0,0.04)] relative overflow-hidden animate-fade-in-up hover:shadow-[0_12px_40px_rgb(0,0,0,0.08)] transition-shadow duration-300">
            <div>
              <h2 className="text-gray-400 text-xs font-bold uppercase tracking-widest mb-4 flex items-center gap-2">
                <Activity className="h-4 w-4 text-primary" /> Overall Score
              </h2>
              <div className="text-8xl md:text-9xl font-bold tracking-tighter text-gray-900 mt-2 leading-none">
                {result.score.toFixed(1)}
              </div>
            </div>
            
            <div className="mt-12">
              <div className={`text-sm font-semibold px-4 py-2 rounded-full inline-flex items-center tracking-wide ${result.score >= 7.0 ? 'bg-green-100 text-green-700' : result.score >= 5.0 ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'}`}>
                {result.score >= 7.0 ? 'HIGH' : result.score >= 5.0 ? 'MEDIUM' : 'LOW'} INTERACTION
              </div>
              <div className="mt-5 text-sm font-medium text-gray-400 flex items-center">
                <MapPin className="inline h-4 w-4 mr-1.5 text-gray-300" />
                {result.location.lat.toFixed(5)}, {result.location.lon.toFixed(5)}
              </div>
            </div>
          </div>

          {/* Map Visualizer (Span 4) */}
          <div className="col-span-1 md:col-span-2 lg:col-span-4 bg-white/80 backdrop-blur-xl border border-white/40 rounded-[2rem] p-4 shadow-[0_8px_30px_rgb(0,0,0,0.04)] h-[350px] lg:h-auto overflow-hidden animate-fade-in-up hover:shadow-[0_12px_40px_rgb(0,0,0,0.08)] transition-shadow duration-300" style={{animationDelay: '0.1s'}}>
             <h2 className="text-gray-400 text-xs font-bold uppercase tracking-widest mb-3 ml-4 mt-2">Spatial Context</h2>
             <div className="w-full h-[calc(100%-3rem)] rounded-2xl overflow-hidden bg-gray-100 relative">
               <MapContainer 
                 center={[result.location.lat, result.location.lon]} 
                 zoom={14} 
                 scrollWheelZoom={false} 
                 style={{ height: '100%', width: '100%' }}
                 className="z-0"
               >
                 {/* Light sleek map tiles */}
                 <TileLayer
                   attribution='&copy; <a href="https://carto.com/">Carto</a>'
                   url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
                 />
                 <Marker position={[result.location.lat, result.location.lon]}>
                   <Popup className="font-sans text-sm font-medium rounded-xl">Target Locked</Popup>
                 </Marker>
                 <ChangeView center={[result.location.lat, result.location.lon]} zoom={14} />
               </MapContainer>
             </div>
          </div>

          {/* Radar Chart (Span 4) */}
          <div className="col-span-1 md:col-span-4 lg:col-span-4 bg-white/80 backdrop-blur-xl border border-white/40 rounded-[2rem] p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] flex flex-col animate-fade-in-up hover:shadow-[0_12px_40px_rgb(0,0,0,0.08)] transition-shadow duration-300" style={{animationDelay: '0.2s'}}>
            <h2 className="text-gray-400 text-xs font-bold uppercase tracking-widest mb-4">Signature Profile</h2>
            <div className="flex-1 w-full min-h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="75%" data={getRadarData(result.features)}>
                  <PolarGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: '#6B7280', fontSize: 12, fontWeight: 500 }} />
                  <Radar name="Metrics" dataKey="A" stroke="#007AFF" strokeWidth={3} fill="#007AFF" fillOpacity={0.15} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', border: 'none', borderRadius: '16px', boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)', padding: '12px 16px' }}
                    itemStyle={{ color: '#1D1D1F', fontWeight: 600 }}
                    cursor={{ stroke: '#E5E7EB', strokeWidth: 1 }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Telemetry Grid (Span 12) */}
          <div className="col-span-1 md:col-span-4 lg:col-span-12 bg-white/40 backdrop-blur-xl border border-white/50 rounded-[2.5rem] p-8 md:p-10 shadow-[0_8px_30px_rgb(0,0,0,0.04)] animate-fade-in-up" style={{animationDelay: '0.3s'}}>
             <h2 className="text-gray-400 text-xs font-bold uppercase tracking-widest mb-8 text-center md:text-left">Telemetry Data Stream</h2>
             <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-7 gap-4 md:gap-6">
                {Object.entries(result.features).map(([key, value]) => {
                  const label = key.replace(/^F\d+_/, '').replace(/_/g, ' ');
                  return (
                    <div key={key} className="bg-white/60 hover:bg-white backdrop-blur-md border border-white/60 rounded-2xl p-5 hover:shadow-lg transition-all duration-300 group cursor-default">
                      <div className="text-[11px] text-gray-500 font-semibold truncate uppercase tracking-wider mb-2 group-hover:text-primary transition-colors" title={label}>{label}</div>
                      <div className="text-2xl font-bold tracking-tight text-gray-900 group-hover:scale-105 transition-transform origin-left">
                        {typeof value === 'number' ? (value % 1 === 0 ? value : value.toFixed(2)) : value}
                      </div>
                    </div>
                  );
                })}
             </div>
          </div>

        </div>
      )}
    </div>
  );
}

export default App;

