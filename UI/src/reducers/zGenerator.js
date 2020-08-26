import { ActionTypes } from '../core/constants'

const initialState = {
  isFetchingInfo: true,
  isFetchingImage: true,
  imgBase64: '',
  modelName: '',
  latentsDimensions: 0,
  randomizeNoise: true,
  psi: 0.9,
  latents: [],
  label: 0
}

const fetchImageStart = (state, action) => ({
  ...state,
  isFetchingImage: true
})


const fetchImageSucceed = (state, action) => ({
  ...state,
  isFetchingImage: false,
  imgBase64: action.imgBase64
})

const fetchImageFail = (state, action) => ({
  ...state,
  isFetchingImage: false
})

const fetchInfoStart = (state, action) => ({
  ...state,
  isFetchingInfo: true
})

const fetchInfoSucceed = (state, action) => {
  const latents = Array(action.data.latentsDimensions).fill(0)
  return {
    ...state,
    isFetchingInfo: false,
    modelName: action.data.modelName,
    latentsDimensions: action.data.latentsDimensions,
    latents
  }
}

const fetchInfoFail = (state, action) => ({
  ...state,
  isFetchingInfo: false
})

const changeLatents = (state, action) => {
  const latents = [...state.latents]
  const { idx, value } = action
  latents[idx] = value
  return {
    ...state,
    latents
  }
}

const changePsi = (state, action) => ({
  ...state,
  psi: action.psi
})

const randomLatents = (state, action) => ({
  ...state,
  latents: [...action.latents]
})

const changeLabel = (state, action) => ({
  ...state,
  label: action.label
})

export default function (state = initialState, action) {
  switch (action.type) {
    case ActionTypes.FETCH_IMAGE_START:
      return fetchImageStart(state, action)
    case ActionTypes.FETCH_IMAGE_SUCCEED:
      return fetchImageSucceed(state, action)
    case ActionTypes.FETCH_IMAGE_FAIL:
      return fetchImageFail(state, action)
    case ActionTypes.FETCH_INFO_START:
      return fetchInfoStart(state, action)
    case ActionTypes.FETCH_INFO_SUCCEED:
      return fetchInfoSucceed(state, action)
    case ActionTypes.FETCH_INFO_FAIL:
      return fetchInfoFail(state, action)
    case ActionTypes.CHANGE_LATENTS:
      return changeLatents(state, action)
    case ActionTypes.CHANGE_PSI:
      return changePsi(state, action)
    case ActionTypes.RANDOM_LATENTS:
      return randomLatents(state, action)
    case ActionTypes.CHANGE_LABEL:
      return changeLabel(state, action)
    default:
      return state
  }
}
