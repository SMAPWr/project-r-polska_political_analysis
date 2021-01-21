import axios from "axios"

const client = axios.create({
    baseURL: 'http://localhost:5000'
})

export async function predict(text) {
    const response = await client.post('/', {text})
    return response.data
}

export async function predict_twitter_user(user) {
    const response = await client.get(`/twitter_user?user=${user}`)
    return response.data
}

export async function predict_hashtag(tag) {
    const response = await client.get(`/hashtag?hashtag=${tag}`)
    return response.data
}