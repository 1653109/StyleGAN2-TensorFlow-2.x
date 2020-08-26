import { createStore, applyMiddleware, compose } from 'redux'
// import {routerMiddleware} from 'react-router-redux'
import thunk from 'redux-thunk'

import rootReducer from './reducers'

const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose

export function configureStore({ initialState, history }) {
  // const router = routerMiddleware(history)
  const middlewares = [thunk]
  const createStoreWithMiddleware = composeEnhancers(applyMiddleware(...middlewares))(createStore)
  const store = createStoreWithMiddleware(rootReducer, initialState)

  return store
}

export default {
  configureStore
}
