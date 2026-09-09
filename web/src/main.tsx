import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles/tokens.css";
import "./styles/pills.css";
import "./styles/chips.css";
import "./styles/modal.css";
import "./styles/timeline.css";
import "./styles/queue.css";
import "./styles/presets.css";
import "./styles/wtmr.css";
import "./styles/models.css";
import "./styles/cast.css";
import "./styles/composer.css";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
