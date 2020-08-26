import axios from 'axios'

import { ActionTypes } from '../../core/constants'
import { encodeFloat32, randn_bm, randn_bm_0_1 } from '../../utils/latentCode'

const fetchImageStart = () => ({
  type: ActionTypes.FETCH_IMAGE_START
})

const fetchImageFail = () => ({
  type: ActionTypes.FETCH_IMAGE_FAIL
})

const fetchImageSucceed = (imgBase64) => ({
  type: ActionTypes.FETCH_IMAGE_SUCCEED,
  imgBase64
})

export const fetchImage = () => async (dispatch, getState) => {
  dispatch(fetchImageStart())

  try {
    const apiUrl = process.env.REACT_APP_API_URL
    const { psi, label, latents } = getState().zGenerator

    const resJson = await axios.get(`${apiUrl}image?psi=${psi}&label=${label}&z=${encodeURIComponent(encodeFloat32(latents))}`)
    const imgBase64 = resJson.data['image_str']
    dispatch(fetchImageSucceed(imgBase64))
  } catch (error) {
    console.log(error)
    dispatch(fetchImageFail())
  }
}

const fetchInfoStart = () => ({
  type: ActionTypes.FETCH_INFO_START
})

const fetchInfoFail = () => ({
  type: ActionTypes.FETCH_INFO_FAIL
})

const fetchInfoSucceed = (data) => ({
  type: ActionTypes.FETCH_INFO_SUCCEED,
  data
})

export const fetchInfo = () => async (dispatch, getState) => {
  dispatch(fetchInfoStart())

  try {
    const apiUrl = process.env.REACT_APP_API_URL

    const infoJson = await axios.get(`${apiUrl}info`)
    const data = {
      modelName: infoJson.data['model'],
      latentsDimensions: infoJson.data['latents_dimensions'],
    }
    dispatch(fetchInfoSucceed(data))
    dispatch(fetchImage())
  } catch (error) {
    dispatch(fetchInfoFail())
  }
}

const changeLatentsAction = (idx, value) => ({
  type: ActionTypes.CHANGE_LATENTS,
  idx,
  value
})

export const changeLatents = (idx, value) => (dispatch, getState) => {
  dispatch(changeLatentsAction(idx, value))
  // dispatch(fetchImage())
}

const changePsiAction = (psi) => ({
  type: ActionTypes.CHANGE_PSI,
  psi
})

export const changePsi = psi => (dispatch, getState) => {
  dispatch(changePsiAction(psi))
  // dispatch(fetchImage())
}

const randomLatentsAction = (latents) => ({
  type: ActionTypes.RANDOM_LATENTS,
  latents
})

export const randomLatents = () => (dispatch, getState) => {
  let latents = [...getState().zGenerator.latents]
  latents = latents.map(() => randn_bm())
  // latents = latents.map(() => randn_bm_0_1())
  dispatch(randomLatentsAction(latents))
  dispatch(fetchImage())
}

const changeLabelAction = label => ({
  type: ActionTypes.CHANGE_LABEL,
  label
})

export const changeLabel = label => (dispatch, getState) => {
  dispatch(changeLabelAction(label))
  dispatch(fetchImage())
}