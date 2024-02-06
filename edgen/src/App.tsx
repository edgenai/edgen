
import "./App.css";

import edgenLogo from "./assets/edgen_logo.svg"

function App() {

  return (
    <div className="container">
      <h1>Welcome to Edgen!</h1>

      <div>
        <img src={edgenLogo} className="logo edgen" alt="Edgen logo" />
      </div>

      <p>GUI coming soon</p>
    </div>
  );
}

export default App;
