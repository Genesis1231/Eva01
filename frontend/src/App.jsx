import { BrowserRouter, Routes, Route } from "react-router-dom";
import Room from "./components/Room";
import CanvasPage from "./components/CanvasPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Room />} />
        <Route path="/canvas" element={<CanvasPage />} />
      </Routes>
    </BrowserRouter>
  );
}
