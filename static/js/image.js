const uploadForm = document.getElementById("upload-form");
const resultDiv = document.getElementById("result");
const predictedClass = document.getElementById("predicted-class");
const cancerProbability = document.getElementById("cancer-probability");
const normalProbability = document.getElementById("normal-probability");
const uploadedImageDiv = document.getElementById("uploaded-image");
const imagePreview = document.getElementById("image-preview");
const errorDiv = document.getElementById("error");

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultDiv.style.display = "none";
  uploadedImageDiv.style.display = "none";
  errorDiv.style.display = "none";

  const formData = new FormData(uploadForm);
  const file = document.getElementById("file").files[0];

  if (file) {
    // Tampilkan gambar yang diunggah
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      uploadedImageDiv.style.display = "block";
    };
    reader.readAsDataURL(file);
  }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (data.success) {
      predictedClass.textContent = data.class;
      cancerProbability.textContent = data.prob_cancer;
      normalProbability.textContent = data.prob_no_cancer;
      resultDiv.style.display = "block";
    } else {
      throw new Error(data.error || "Gagal memproses gambar.");
    }
  } catch (error) {
    errorDiv.textContent = error.message;
    errorDiv.style.display = "block";
  }
});
