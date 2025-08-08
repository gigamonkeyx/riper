import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import L from 'leaflet';

const pin = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41], iconAnchor: [12, 41]
});

const customers = [
  { name: 'Rocking Horse Bakery', coords: [48.6496, -119.5467] },
  { name: 'Sugar Quail', coords: [48.7032, -119.4350] },
  { name: 'Copper Eagle', coords: [48.6763, -119.5458] },
  { name: 'Hometown Pizza', coords: [48.7061, -119.4399] },
];

const suppliers = [
  { name: 'Bluebird Grain Farms', coords: [47.5444, -120.4518] },
  { name: 'Whitestone Orchards', coords: [48.6285, -119.5887] },
  { name: 'Billy\'s Gardens', coords: [48.6021, -119.4421] },
  { name: 'Double S Meats', coords: [48.4752, -119.5041] },
];

const base = [48.7052, -119.4370]; // Tonasket approx

export default function MapSection({ kind = 'customers' }) {
  const points = kind === 'customers' ? customers : suppliers;
  const routes = points.map(p => [base, p.coords]);

  return (
    <div className="map-container">
      <MapContainer center={base} zoom={10} scrollWheelZoom={false} style={{height:'100%', width:'100%'}}>
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <Marker position={base} icon={pin}><Popup>Tonasket Base</Popup></Marker>
        {points.map((p, idx) => (
          <Marker key={idx} position={p.coords} icon={pin}>
            <Popup>{p.name}</Popup>
          </Marker>
        ))}
        {routes.map((r, idx) => (
          <Polyline key={idx} positions={r} color={kind === 'customers' ? '#203a43' : '#6b7e3b'} />
        ))}
      </MapContainer>
    </div>
  );
} 