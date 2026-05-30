// WebSocketService.js - Handles WebSocket communication with the EVA backend
import config from "../config";

class WebSocketService {
  constructor() {
    this.socket = null;
    this.clientId = this.generateClientId();
    this.onMessageCallback = null;
    this.onImageRequestCallback = null;
    this.onConnectionStatusCallback = null;
    this.isConnected = false;
    this.connectionPromise = null;
    this.sessionId = this.generateSessionId();
    this.baseUrl = config.websocket.baseUrl;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = config.websocket.reconnectAttempts;
  }

  generateClientId() {
    return "client_" + Math.random().toString(36).substring(2, 15);
  }

  generateSessionId() {
    return "session_" + Math.random().toString(36).substring(2, 15);
  }

  connect() {
    if (
      this.socket &&
      (this.socket.readyState === WebSocket.CONNECTING ||
        this.socket.readyState === WebSocket.OPEN)
    ) {
      console.log(
        "WebSocket already connecting or connected, reusing connection",
      );
      return Promise.resolve();
    }

    if (!this.connectionPromise) {
      this.connectionPromise = new Promise((resolve, reject) => {
        try {
          const url = `${this.baseUrl}/ws/${this.clientId}`;

          console.log(`Connecting to WebSocket at ${url}`);

          this.updateConnectionStatus("connecting");

          this.socket = new WebSocket(url);

          this.socket.onopen = () => {
            this.isConnected = true;
            this.reconnectAttempts = 0;
            console.log("WebSocket connection established");
            this.updateConnectionStatus("connected");

            if (this.onImageRequestCallback) {
              console.log("Requesting initial image capture after connection");
              setTimeout(() => {
                this.onImageRequestCallback("frontImage");
              }, 500);
            } else {
              console.error(
                "No image request callback registered! The backend cannot receive the initial observation.",
              );
            }

            resolve();
          };

          this.socket.onmessage = (event) => {
            this.handleMessage(event.data);
          };

          this.socket.onclose = (event) => {
            this.isConnected = false;
            console.log(`WebSocket connection closed with code ${event.code}`);
            this.connectionPromise = null;
            this.updateConnectionStatus("disconnected");

            if (event.code !== 1000 && event.code !== 1001) {
              if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;

                const delay = config.websocket.reconnectInterval || 3000;
                console.log(
                  `Attempting to reconnect in ${delay / 1000} seconds (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`,
                );
                this.updateConnectionStatus(
                  "reconnecting",
                  this.reconnectAttempts,
                );

                setTimeout(() => {
                  this.connect().catch((err) => {
                    console.error("Failed to reconnect:", err);
                    this.updateConnectionStatus("failed");
                  });
                }, delay);
              } else {
                console.error("Maximum reconnection attempts reached.");
                this.updateConnectionStatus("failed");
              }
            }
          };

          this.socket.onerror = (error) => {
            console.error("WebSocket error:", error);
            this.updateConnectionStatus("error");
            reject(error);
          };
        } catch (error) {
          console.error("Failed to create WebSocket connection:", error);
          this.connectionPromise = null;
          this.updateConnectionStatus("error");
          reject(error);
        }
      });
    }

    return this.connectionPromise;
  }

  updateConnectionStatus(status, data = null) {
    if (this.onConnectionStatusCallback) {
      this.onConnectionStatusCallback(status, data);
    }
  }

  async handleMessage(data) {
    try {
      const parsedData = JSON.parse(data);

      console.log("Received WebSocket message:", parsedData);

      if (parsedData.type === "receive_start") {
        if (parsedData.session_id) {
          console.log(
            "Received session ID from backend:",
            parsedData.session_id,
          );
          this.sessionId = parsedData.session_id;
        }

        if (this.onImageRequestCallback) {
          console.log("Sending initial observation to start conversation");
          setTimeout(() => {
            this.onImageRequestCallback("frontImage");
          }, 300);
        }
        return;
      }

      if (parsedData.type && parsedData.type.startsWith("validation-")) {
        console.log(
          `Message validation: ${parsedData.type} - ${parsedData.content}`,
        );
        return;
      }

      if (parsedData.type === "request_image") {
        if (this.onImageRequestCallback) {
          console.log(
            `Backend requested an image: ${parsedData.imageType || "frontImage"}`,
          );
          this.onImageRequestCallback(parsedData.imageType || "frontImage");
        }
        return;
      }

      if (parsedData.session_id) {
        this.sessionId = parsedData.session_id;
      }

      if (Array.isArray(parsedData)) {
        console.log("Processing array of messages, length:", parsedData.length);
        for (const item of parsedData) this.emitMessage(item);
      } else if (parsedData.type) {
        this.emitMessage(parsedData);
      }
    } catch (error) {
      console.error("Error processing WebSocket message:", error, data);
    }
  }

  // Forward one message to the consumer, normalizing audio into a playable
  // download URL; everything else passes through untouched.
  emitMessage(item) {
    if (!this.onMessageCallback) return;
    if (item.type === "audio") {
      this.onMessageCallback({
        type: "audio",
        content: this.formatAudioUrl(item.content),
        text: item.text,
        session_id: this.sessionId,
      });
    } else {
      this.onMessageCallback(item);
    }
  }

  formatAudioUrl(audioPath) {
    const baseUrl = config.api.baseUrl;

    let audioUrl;
    if (audioPath.startsWith("http://") || audioPath.startsWith("https://")) {
      audioUrl = audioPath;
    } else if (audioPath.includes("/download/")) {
      audioUrl = `${baseUrl}${audioPath.startsWith("/") ? "" : "/"}${audioPath}`;
    } else if (audioPath.includes("audio/")) {
      const parts = audioPath.split("audio/");
      const filename = parts[1]?.split("?")[0];
      audioUrl = `${baseUrl}/download/audio/${filename}`;
    } else {
      audioUrl = `${baseUrl}/download/${audioPath.startsWith("/") ? audioPath.substring(1) : audioPath}`;
    }

    if (!audioUrl.includes("session_id=")) {
      audioUrl = `${audioUrl}${audioUrl.includes("?") ? "&" : "?"}session_id=${this.sessionId}`;
    }

    return audioUrl;
  }

  // The backend only processes a turn after an "over" marker; every send* path
  // ends with one.
  sendOver() {
    const overPayload = {
      type: "over",
      session_id: this.sessionId,
      content: "done",
    };
    console.log('Sending "over" message to trigger backend processing');
    this.socket.send(JSON.stringify(overPayload));
  }

  async sendMessage(message) {
    try {
      await this.connect();

      const payload = {
        type: "text",
        session_id: this.sessionId,
        content: message,
      };

      console.log("Sending text message to WebSocket:", payload);
      this.socket.send(JSON.stringify(payload));

      this.sendOver();
    } catch (error) {
      console.error("Error sending message:", error);
      throw error;
    }
  }

  async sendAudio(audioBlob) {
    try {
      await this.connect();

      if (!audioBlob || audioBlob.size === 0) {
        throw new Error("Invalid audio blob: empty or zero size");
      }

      console.log(
        `Audio blob from recorder: size=${audioBlob.size}, type=${audioBlob.type}`,
      );

      if (audioBlob.type !== "audio/mp3" && audioBlob.type !== "audio/mpeg") {
        console.warn(
          `Received non-MP3 audio (${audioBlob.type}). AudioRecorder should convert to MP3 first.`,
        );
      }

      const base64Audio = await this.blobToBase64(audioBlob);

      if (!base64Audio) {
        throw new Error("Failed to convert audio to base64");
      }

      if (base64Audio.length < 100) {
        console.warn(
          "Audio base64 string seems very short, may not contain valid audio data",
        );
        throw new Error("Audio data appears to be invalid (too small)");
      }

      const payload = {
        type: "audio",
        session_id: this.sessionId,
        content: base64Audio,
      };

      console.log("Sending audio data via WebSocket");
      this.socket.send(JSON.stringify(payload));

      this.sendOver();
    } catch (error) {
      console.error("Error sending audio:", error);
      throw error;
    }
  }

  async sendImage(imageBlob, isFrontImage = true) {
    console.log(
      "sendImage called with blob:",
      imageBlob ? `${imageBlob.size} bytes` : "null",
    );

    try {
      await this.connect();

      const base64Image = await this.blobToBase64(imageBlob);

      const payload = {
        type: isFrontImage ? "frontImage" : "backImage",
        session_id: this.sessionId,
        content: base64Image,
      };

      console.log(
        `Sending ${isFrontImage ? "frontImage" : "backImage"} to backend with session_id: ${this.sessionId}`,
      );
      this.socket.send(JSON.stringify(payload));

      this.sendOver();
    } catch (error) {
      console.error("Failed sending image:", error);
      throw error;
    }
  }

  blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      if (!blob || blob.size === 0) {
        console.error("Invalid blob provided to blobToBase64");
        reject(new Error("Invalid blob"));
        return;
      }

      const reader = new FileReader();

      reader.onloadend = () => {
        try {
          const base64String = reader.result.split(",")[1];

          if (!base64String) {
            console.error("Failed to extract base64 string from reader result");
            reject(new Error("Failed to extract base64 string"));
            return;
          }

          resolve(base64String);
        } catch (err) {
          console.error("Error processing base64 conversion:", err);
          reject(err);
        }
      };

      reader.onerror = (error) => {
        console.error("Error reading blob:", error);
        reject(error);
      };

      reader.readAsDataURL(blob);
    });
  }

  setMessageCallback(callback) {
    this.onMessageCallback = callback;
  }

  setImageRequestCallback(callback) {
    this.onImageRequestCallback = callback;
    console.log(
      "Image request callback " + (callback ? "registered" : "cleared"),
    );
  }

  setConnectionStatusCallback(callback) {
    this.onConnectionStatusCallback = callback;
    if (callback) {
      if (this.isConnected) {
        callback("connected");
      } else if (
        this.socket &&
        this.socket.readyState === WebSocket.CONNECTING
      ) {
        callback("connecting");
      } else {
        callback("disconnected");
      }
    }
  }

  disconnect() {
    if (this.socket && this.isConnected) {
      this.socket.close();
      this.isConnected = false;
      this.connectionPromise = null;
      this.updateConnectionStatus("disconnected");
    }
  }
}

const webSocketService = new WebSocketService();
export default webSocketService;
