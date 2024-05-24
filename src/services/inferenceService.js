const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat()
 
        const classes = ['Non-cancer','Cancer'];
 
        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;
        const classResult = confidenceScore > 50 ? 1 : 0;
        const label = classes[classResult];
 
        let suggestion;
 
        if(label === 'Non-cancer') {
            suggestion = "Bukan Kanker."
        }
 
        if(label === 'Cancer') {
            suggestion = "Segera konsultasi dengan dokter terdekat untuk meminimalisasi penyebaran kanker."
        }
 
        return { label, suggestion };
    } catch (error) {
        throw new InputError(`Terjadi kesalahan dalam melakukan prediksi`,400);
    }
}
 
module.exports = predictClassification;