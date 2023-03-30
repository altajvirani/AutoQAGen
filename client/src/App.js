import React, { useEffect, useState, useRef } from 'react'
import 'bootstrap/dist/css/bootstrap.css'
import Container from 'react-bootstrap/Container'
import Row from 'react-bootstrap/Row'
import Col from 'react-bootstrap/Col'
import Form from 'react-bootstrap/Form'
import FormLabel from 'react-bootstrap/FormLabel'
import Select from 'react-select'
import makeAnimated from 'react-select/animated'
import FormControl from 'react-bootstrap/FormControl'
import Button from 'react-bootstrap/Button'
import Tab from 'react-bootstrap/Tab'
import Tabs from 'react-bootstrap/Tabs'
import FormSelect from 'react-bootstrap/esm/FormSelect'
import Toast from './Toast'
import './App.css'

function App() {
  const inputText = useRef(null)
  const noOfQns = useRef(null)
  const dwnldOp = useRef()
  const [qType, setQType] = useState([])
  const [activeTab, setActiveTab] = useState(null)
  const [whQnsAns, setWhQnsAns] = useState()
  const [gapQnsAns, setGapQnsAns] = useState()
  const [mcqQnsAns, setMcqQnsAns] = useState()
  const [boolQnsAns, setBoolQnsAns] = useState()
  const [subjQnsAns, setSubjQnsAns] = useState()
  const animatedComponents = makeAnimated()
  const [toast, setToast] = useState(null)

  const qTypeList = [{
    label: 'Single Answer',
    key: 'single_answer',
    value: 'single_answer',
    state: whQnsAns
  }, {
    label: 'Fill-ups',
    key: 'gap_fill',
    value: 'gap_fill',
    state: gapQnsAns
  }, {
    label: 'MCQ',
    key: 'mcq',
    value: 'mcq',
    state: mcqQnsAns
  }, {
    label: 'Yes/No',
    key: 'boolean',
    value: 'boolean',
    state: boolQnsAns
  }, {
    label: 'Full Sentence',
    key: 'subjective',
    value: 'subjective',
    state: subjQnsAns
  }]

  const styles = {
    valueContainer: (provided) => ({
      ...provided,
      flexWrap: 'nowrap',
    }),
    input: (provided) => ({
      ...provided,
      maxWidth: '100%',
      whiteSpace: 'nowrap'
    }),
    control: (provided, state) => ({
      ...provided,
      border: state.isFocused ? '1px solid #92c7ff' : '1px solid #C7D1DF',
      borderRadius: '0.5rem',
      boxShadow: state.isFocused ? '0 0 0 0.25rem rgba(0, 123, 255, 0.25)' : null,
      '&:hover': {
        borderColor: state.isFocused ? '#92c7ff' : '#C7D1DF'
      }
    })
  }

  const handleQType = e => {
    const keys = e.map(obj => obj.key)
    setQType(keys)
  }

  useEffect(() => {
    if (subjQnsAns) setActiveTab('subjective')
    if (boolQnsAns) setActiveTab('boolean')
    if (mcqQnsAns) setActiveTab('mcq')
    if (gapQnsAns) setActiveTab('gap_fill')
    if (whQnsAns) setActiveTab('single_answer')
  }, [whQnsAns, gapQnsAns, mcqQnsAns, boolQnsAns, subjQnsAns])

  const handleSubmit = async () => {
    if (inputText.current.value && noOfQns.current.value && qType.length) {
      const response = await fetch('https://autoqa-gen-backend.onrender.com/send_doc', {
      // const response = await fetch('/send_doc', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          document: inputText.current.value,
          qn_type: qType,
          no_of_qns: noOfQns.current.value
        })
      })
      const data = await response.json()
      console.log(data)
      setWhQnsAns(data['single_answer'] ?? '')
      setGapQnsAns(data['gap_fill'] ?? '')
      setMcqQnsAns(data['mcq'] ?? '')
      setBoolQnsAns(data['boolean'] ?? '')
      setSubjQnsAns(data['subjective'] ?? '')
      setToast(null)
    } else {
      setToast({
        'type': 'default',
        'title': 'Please fill in all fields.',
        'duration': '2000',
        'position': 'top center'
      })
    }
  }

  const capitalizeStr = str => {
    return (
      str.charAt(0).toUpperCase() + str.slice(1).toLowerCase()
    );
  };


  const dwnldCSV = () => {
    let csvContent = "data:text/csvcharset=utf-8,"
    let header = ["Sr. No.", "Question Type", "Question"]
    let optionsCount = 0
    if (mcqQnsAns) {
      mcqQnsAns.map(val =>
        optionsCount = val.options.length > optionsCount ? val.options.length : optionsCount
      )
      for (let i = 1; i <= optionsCount; i++)
        header.push("Option " + i)
    }
    header.push("Answer")
    csvContent += header.join(",") + "\n"
    const addDataRows = (qnsAns, type) => {
      if (!qnsAns) return
      qnsAns.forEach((qa, index) => {
        let row = [index + 1, type, qa.question]
        if (mcqQnsAns && qa.options) {
          qa.options.forEach(opt =>
            row.push(opt)
          )
        } else {
          for (let i = 0; i < optionsCount; i++)
            row.push('-')
        }
        row.push(qa.answer)
        csvContent += row.join(",") + "\n"
      })
    }
    addDataRows(whQnsAns, "Wh-type")
    addDataRows(gapQnsAns, "Gap-Fill")
    addDataRows(mcqQnsAns, "MCQ")
    addDataRows(boolQnsAns, "True/False")
    addDataRows(subjQnsAns, "Subjective")
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", "qns_ans.csv")
    document.body.appendChild(link)
    if (whQnsAns || gapQnsAns || mcqQnsAns || boolQnsAns || subjQnsAns)
      link.click()
    document.body.removeChild(link)
  }

  const dwnldExcel = () => {
    console.log('excel')
  }

  const handleDwnld = () => {
    if (dwnldOp.current.selectedIndex === 1)
      dwnldCSV()
    if (dwnldOp.current.selectedIndex === 2)
      dwnldExcel()
    if (dwnldOp.current.selectedIndex === 3)
      console.log('pdf')
  }

  return (
    <Container fluid className='qa-form-container'>
      <div className='qa-form-header'>
        <p className='header'>AutoQAGen</p>
      </div>
      <Row className='qa-form-body'>
        <Col xs={12} sm={12} md={7} lg={7} xl={7} col={7} className='qa-form-section-1'>
          <div className='qa-form-section-container-1'>
            <Tabs className='qa-ans-tabs mb-3' activeKey={activeTab} onSelect={e => setActiveTab(e)}>
              {
                qTypeList.map(value => {
                  const disabled = !value.state
                  return (
                    <Tab
                      key={value.key}
                      eventKey={value.key}
                      title={value.label}
                      disabled={disabled} />
                  )
                })}
            </Tabs>
            <Container fluid className='qa-ans-cont'>
              {whQnsAns ||
                gapQnsAns ||
                mcqQnsAns ||
                boolQnsAns ||
                subjQnsAns
                ? <div className='qa-ans-table-div'>
                  <table className='qa-ans-table'>
                    <thead>
                      <tr>
                        <th style={{
                          width: '1%'
                        }}>No.</th>
                        {activeTab === 'single_answer' || activeTab === 'gap_fill'
                          ? <th style={{
                            width: '70%'
                          }}>Question</th>
                          : activeTab === 'mcq' || activeTab === 'subjective'
                            ? <th style={{
                              width: '45%'
                            }}>Question</th>
                            : activeTab === 'boolean'
                              ? <th style={{
                                width: '80%'
                              }}>Question</th>
                              : null}
                        <th>Answer</th>
                      </tr>
                    </thead>
                    <tbody>
                      {!activeTab || activeTab === 'single_answer' ?
                        (whQnsAns
                          ? whQnsAns.map((val, key) => {
                            return (
                              <tr key={++key} className='qa-ans-table-row'>
                                <td>{++key}</td>
                                <td>{capitalizeStr(val.question)}</td>
                                <td>{capitalizeStr(val.answer)}</td>
                              </tr>)
                          }) : null)
                        : activeTab === 'gap_fill'
                          ? (gapQnsAns
                            ? gapQnsAns.map((val, key) => {
                              return (
                                <tr key={++key} className='qa-ans-table-row'>
                                  <td>{++key}</td>
                                  <td>{capitalizeStr(val.question)}</td>
                                  <td>{capitalizeStr(val.answer)}</td>
                                </tr>)
                            }) : null)
                          : activeTab === 'mcq'
                            ? (mcqQnsAns
                              ? mcqQnsAns.map((val, key) => {
                                return (
                                  <tr key={++key} className='qa-ans-table-row'>
                                    <td>{++key}</td>
                                    <td>{capitalizeStr(val.question)}</td>
                                    {val.options.map((ans, key) => {
                                      return (
                                        ans === val.answer
                                          ? <td key={key} style={{ fontWeight: '900' }}>{capitalizeStr(ans)}</td>
                                          : <td key={key}>{capitalizeStr(ans)}</td>
                                      )
                                    })}
                                  </tr>)
                              }) : null)
                            : activeTab === 'boolean'
                              ? (boolQnsAns
                                ? boolQnsAns.map((val, key) => {
                                  return (
                                    <tr key={++key} className='qa-ans-table-row'>
                                      <td>{++key}</td>
                                      <td>{capitalizeStr(val.question)}</td>
                                      <td>{capitalizeStr(val.answer)}</td>
                                    </tr>)
                                }) : null)
                              : activeTab === 'subjective'
                                ? (subjQnsAns
                                  ? subjQnsAns.map((val, key) => {
                                    return (
                                      <tr key={++key} className='qa-ans-table-row'>
                                        <td>{++key}</td>
                                        <td>{capitalizeStr(val.question)}</td>
                                        <td>{capitalizeStr(val.answer)}</td>
                                      </tr>
                                    )
                                  }) : null)
                                : null
                      }
                    </tbody>
                  </table>
                </div>
                : null}
            </Container>
            <Container fluid className='qa-dwnld-cont'>
              <FormSelect className='qa-dwnld-select' ref={dwnldOp}>
                <option key={0} disabled selected={true}>Download as...</option>
                {[{
                  'text': 'CSV',
                  'val': 'csv'
                }, {
                  'text': 'Excel',
                  'val': 'excel'
                }, {
                  'text': 'PDF',
                  'val': 'pdf'
                }].map((v, k) => {
                  return (
                    <option
                      key={k}
                      value={v.val}
                      className='qa-dwnld-select-option'>{v.text}
                    </option>)
                })}
              </FormSelect>
              <Button className='qa-dwnld-button' onClick={() => handleDwnld()}>Download</Button>
            </Container>
          </div>
        </Col>
        <Col xs={12} sm={12} md={5} lg={5} xl={5} col={5} className='qa-form-section-2'>
          <Container fluid className='qa-form-section-container-2'>
            <Form className='qa-form'>
              <FormLabel className='label'>Enter text</FormLabel>
              <textarea className='form-control qa-textarea'
                placeholder='Enter text...'
                required={true}
                autoFocus={true}
                ref={inputText} />
              <FormLabel className='label'>Question Type</FormLabel>
              <Select
                className='qa-qtype'
                closeMenuOnSelect={false}
                components={animatedComponents}
                isMulti
                isSearchable
                options={qTypeList}
                styles={styles}
                onChange={(e) => handleQType(e)} />
              <FormLabel className='label'>No. of Questions</FormLabel>
              <FormControl
                className='qa-form-control'
                type='number'
                min='1'
                placeholder='Eg. 1'
                pattern='^[0-9]+'
                required={true}
                ref={noOfQns} />
              <Button
                className='qa-form-button'
                variant='primary'
                type='button'
                onClick={() => handleSubmit()}>Submit
              </Button>
            </Form>
          </Container>
        </Col>
      </Row>
      {toast
        ? <Toast
          type={toast.type}
          title={toast.title}
          duration={toast.duration}
          position={toast.position}
          showToast={setToast} />
        : null}
    </Container>
  )
}

export default App