#include <Arduino.h>
#include "driver/i2s.h"

// I2S configuration
#define I2S_SAMPLE_RATE (16000)                         // Sample rate of the I2S microphone
#define I2S_CHANNEL_NUM (1)                             // Mono audio channel
#define I2S_BITS_PER_SAMPLE (I2S_BITS_PER_SAMPLE_16BIT) // 16-bit audio samples

void setup()
{
  Serial.begin(115200);

  i2s_config_t i2s_config = {
      .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX), // I2S receive mode
      .sample_rate = I2S_SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // Mono channel
      .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S_MSB),
      .intr_alloc_flags = 0, // Default interrupt allocation
      .dma_buf_count = 8,    // Number of DMA buffers
      .dma_buf_len = 64,     // Size of each DMA buffer
      .use_apll = false      // Use the internal APLL (Audio PLL)
  };

  // Install and configure the I2S driver
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);

  // Set pins for I2S
  i2s_pin_config_t pin_config = {
      .bck_io_num = 26,
      .ws_io_num = 22,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = 25};
  i2s_set_pin(I2S_NUM_0, &pin_config);

  // Configure I2S input format
  i2s_set_clk(I2S_NUM_0, I2S_SAMPLE_RATE, I2S_BITS_PER_SAMPLE, I2S_CHANNEL_MONO);
}

void loop()
{
  // Read data from the I2S microphone
  size_t bytes_read;
  int16_t data[64]; // Adjust the size based on your needs
  i2s_read(I2S_NUM_0, data, sizeof(data), &bytes_read, portMAX_DELAY);

  // Process the audio data and send to Serial Plotter
  for (size_t i = 0; i < 64; i++)
  {
    Serial.print(data[i]);
    Serial.print(" ");
  }
  Serial.println();

  delay(100); // Adjust delay based on your application's needs
}
