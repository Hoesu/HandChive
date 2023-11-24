import React from 'react';
import ImgUpload from '../src/component/ImgUpload';
import styled from 'styled-components';
import Button from './component/Button';

// 전체적인 레이아웃을 담당하는 코드
const Wrapper = styled.div`
    padding: 16px;
    //width: calc(100% - 50px);
    display: flex;
    flex-direction: column;
    //align-items: center;
    justify-content: center;
    background-color: aliceblue;
`;


// 제목 담당
const MainTitleText = styled.p`
    font-size: 37px;
    font-weight: bold;
    text-align: center;
    margin-bottom: -10px;
`;

// 소제목
const SubTitleText = styled.p`
    font-size: 15px;
    text-align: center;
`;

// 업로드 컴포넌트 정렬
const UploadContainer = styled.div`
    margin-top: 40px;
    display: flex;
    justify-content: center;
    align-items: center;
`;

const App = () => {
  const handleFileSelect = (file) => {
    // 선택한 파일을 처리하는 로직을 추가할 수 있습니다.
    console.log('Selected file:', file);
    // 여기서 파일을 업로드하거나 다른 작업을 수행할 수 있습니다.
  };

  const handleButtonClick = () => {
    console.log("Button clicked!");
    //navigate('/emotion-analysis'); 페이지 이동시 사용
  };

  return (
  <div>
    <Wrapper>
        <MainTitleText>📝 HandChive</MainTitleText>
        <SubTitleText>내 손으로 직접 그리는 문서</SubTitleText>
    </Wrapper>
    <UploadContainer>
      <ImgUpload onFileSelect={handleFileSelect} />
    </UploadContainer>
    <Button
      title = "변환하기"
      onClick={handleButtonClick} />
  </div>
  );
};

export default App;
